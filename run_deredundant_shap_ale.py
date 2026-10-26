from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx"
    if (BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx").exists()
    else BASE_DIR / "Cleaned_Durian_Data_ML.xlsx"
)

OUTPUT_DIR = BASE_DIR / "results" / "output_reduced_shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.01,
    "reg_lambda": 5,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0
}

OUTCOME = "yieldha"

FEATURES = [
    "totalwaterha",
    "pesticidesha",
    "treeageyears",
    "productivetreeha",
    "productive_tree_share",
    "gapscoreoverall",
    "ipm",
    "soilimprovement",
    "pdt",
    "plantmonthong",
    "plantkradumthong",
    "plantkanyao",
    "experienceyears",
    "training",
    "extensionvisits",
    "education",
    "age",
    "familymember",
    "groupmember",
    "creditaccess",
    "ph",
    "om",
    "p",
    "k",
    "ec",
    "soilfertilityindex"
]

BLOCKS = {
    "management_inputs": [
        "totalwaterha",
        "pesticidesha",
        "gapscoreoverall",
        "ipm",
        "soilimprovement",
        "pdt",
        "training",
        "extensionvisits",
        "creditaccess"
    ],
    "orchard_structure": [
        "treeageyears",
        "productivetreeha",
        "productive_tree_share",
        "plantmonthong",
        "plantkradumthong",
        "plantkanyao"
    ],
    "farmer_characteristics": [
        "experienceyears",
        "education",
        "age",
        "familymember",
        "groupmember"
    ],
    "soil_context": [
        "ph",
        "om",
        "p",
        "k",
        "ec",
        "soilfertilityindex"
    ]
}

NUMERIC_RAW = [
    "yieldha",
    "totalwaterha",
    "pesticidesha",
    "treeageyears",
    "productivetreeha",
    "treedensityha",
    "experienceyears",
    "extensionvisits",
    "age",
    "education",
    "familymember",
    "ph",
    "om",
    "p",
    "k",
    "ec",
    "soilfertilityindex",
    "pdt",
    "gapscoreoverall"
]

df = pd.read_excel(DATA_PATH, sheet_name="Data")
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(" ", "", regex=False)
    .str.replace("_", "", regex=False)
    .str.lower()
)

for col in NUMERIC_RAW:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

df["productive_tree_share"] = np.where(
    df["treedensityha"] > 0,
    df["productivetreeha"] / df["treedensityha"],
    np.nan
)

missing_features = [
    x for x in FEATURES
    if x not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features in data: {missing_features}"
    )

analysis = df[[OUTCOME] + FEATURES].copy()

for col in [OUTCOME] + FEATURES:
    analysis[col] = pd.to_numeric(
        analysis[col],
        errors="coerce"
    )

analysis = analysis.dropna(
    subset=[OUTCOME]
).reset_index(drop=True)

X_raw = analysis[FEATURES]
y = analysis[OUTCOME].to_numpy()

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X_raw),
    columns=FEATURES,
    index=X_raw.index
)

model = XGBRegressor(**XGB_PARAMS)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap_kg_ha": mean_abs_shap,
    "feature_mean": X.mean().values,
    "feature_sd": X.std().values,
    "shap_feature_correlation": [
        np.corrcoef(
            X.iloc[:, i],
            shap_values[:, i]
        )[0, 1]
        for i in range(len(FEATURES))
    ]
}).sort_values(
    "mean_abs_shap_kg_ha",
    ascending=False
).reset_index(drop=True)

importance["shap_share_percent"] = (
    100
    * importance["mean_abs_shap_kg_ha"]
    / importance["mean_abs_shap_kg_ha"].sum()
)

block_rows = []

for block_name, block_features in BLOCKS.items():
    included = [
        x for x in block_features
        if x in importance["feature"].tolist()
    ]

    total = importance.loc[
        importance["feature"].isin(included),
        "mean_abs_shap_kg_ha"
    ].sum()

    block_rows.append({
        "block": block_name,
        "features": "; ".join(included),
        "n_features": len(included),
        "mean_abs_shap_kg_ha": total
    })

block_importance = pd.DataFrame(block_rows)

block_importance["shap_share_percent"] = (
    100
    * block_importance["mean_abs_shap_kg_ha"]
    / block_importance["mean_abs_shap_kg_ha"].sum()
)

importance.to_csv(
    OUTPUT_DIR / "deredundant_shap_feature_importance.csv",
    index=False
)

block_importance.to_csv(
    OUTPUT_DIR / "deredundant_shap_block_importance.csv",
    index=False
)

with open(
    OUTPUT_DIR / "deredundant_shap_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "outcome": OUTCOME,
            "outcome_units": "kg/ha",
            "observations": int(len(X)),
            "predictors": FEATURES,
            "model_parameters": XGB_PARAMS,
            "interpretation": (
                "SHAP values are model-based predictive "
                "attributions, not causal effects."
            )
        },
        f,
        indent=4
    )

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "deredundant_shap_summary_bar_top20.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "deredundant_shap_beeswarm_top20.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

block_plot = block_importance.sort_values(
    "mean_abs_shap_kg_ha"
)

plt.figure(figsize=(9, 5))
plt.barh(
    block_plot["block"],
    block_plot["mean_abs_shap_kg_ha"],
    color="#2a6f97"
)
plt.xlabel("Aggregate mean absolute SHAP value (kg/ha)")
plt.ylabel("")
plt.title("Feature-family attribution: de-redundant model")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "deredundant_shap_block_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

def one_dimensional_ale(model_object, X_data, feature, bins=20):
    values = X_data[feature].to_numpy()

    edges = np.unique(
        np.quantile(
            values,
            np.linspace(0, 1, bins + 1)
        )
    )

    if len(edges) < 3:
        raise ValueError(
            f"{feature}: insufficient variation for ALE."
        )

    records = []

    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper <= lower:
            continue

        if upper == edges[-1]:
            mask = (
                (values >= lower)
                & (values <= upper)
            )
        else:
            mask = (
                (values >= lower)
                & (values < upper)
            )

        n_bin = int(mask.sum())

        if n_bin == 0:
            continue

        X_low = X_data.loc[mask].copy()
        X_high = X_data.loc[mask].copy()

        X_low[feature] = lower
        X_high[feature] = upper

        local_effect = (
            model_object.predict(X_high)
            - model_object.predict(X_low)
        )

        records.append({
            "feature": feature,
            "lower": lower,
            "upper": upper,
            "midpoint": (lower + upper) / 2,
            "n": n_bin,
            "mean_local_effect": local_effect.mean()
        })

    ale = pd.DataFrame(records)

    ale["ale_uncentered"] = (
        ale["mean_local_effect"].cumsum()
    )

    weights = ale["n"] / ale["n"].sum()

    ale["ale_centered_kg_ha"] = (
        ale["ale_uncentered"]
        - (
            ale["ale_uncentered"]
            * weights
        ).sum()
    )

    return ale

ALE_FEATURES = [
    "treeageyears",
    "totalwaterha",
    "experienceyears",
    "productivetreeha"
]

all_ale = []

for feature in ALE_FEATURES:
    ale = one_dimensional_ale(
        model_object=model,
        X_data=X,
        feature=feature,
        bins=20
    )

    all_ale.append(ale)

    ale.to_csv(
        OUTPUT_DIR / f"deredundant_ale_{feature}.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        ale["midpoint"],
        ale["ale_centered_kg_ha"],
        marker="o",
        color="#2a6f97",
        linewidth=2
    )
    plt.axhline(
        0,
        color="black",
        linewidth=1,
        linestyle="--"
    )
    plt.xlabel(feature)
    plt.ylabel(
        "ALE: deviation from mean prediction (kg/ha)"
    )
    plt.title(
        f"Accumulated local effect: {feature}"
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"deredundant_ale_{feature}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

ale_all = pd.concat(
    all_ale,
    ignore_index=True
)

ale_all.to_csv(
    OUTPUT_DIR / "deredundant_ale_all.csv",
    index=False
)

print("\n" + "=" * 85)
print("DE-REDUNDANT SHAP RESULTS")
print("=" * 85)

print("\nTop 20 features:")
print(
    importance.head(20)
    .round(6)
    .to_string(index=False)
)

print("\nBlock attribution:")
print(
    block_importance
    .round(6)
    .to_string(index=False)
)

print("\nDE-REDUNDANT ALE RESULTS")
print("=" * 85)

for feature in ALE_FEATURES:
    print(f"\n{feature}")
    print(
        ale_all.loc[
            ale_all["feature"] == feature,
            [
                "midpoint",
                "n",
                "ale_centered_kg_ha"
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

print("\nFiles saved:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(path.name)

print("\nCompleted.")