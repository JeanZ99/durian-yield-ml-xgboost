from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx"
    if (BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx").exists()
    else BASE_DIR / "Cleaned_Durian_Data_ML.xlsx"
)

OUTPUT_DIR = BASE_DIR / "results" / "output_reduced_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

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
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0
}

df = pd.read_excel(DATA_PATH, sheet_name="Data")
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(" ", "", regex=False)
    .str.replace("_", "", regex=False)
    .str.lower()
)

numeric_raw = [
    "yieldha",
    "totalwaterha",
    "totalfertha",
    "totalorgfertha",
    "productivetreeha",
    "treedensityha",
    "treeageyears",
    "pesticidesha",
    "experienceyears",
    "extensionvisits",
    "age",
    "education",
    "familymember",
    "ph",
    "om",
    "n",
    "p",
    "k",
    "ec",
    "soilfertilityindex",
    "pdt",
    "gapscoreoverall"
]

for col in numeric_raw:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

df["water_per_productive_tree"] = np.where(
    df["productivetreeha"] > 0,
    df["totalwaterha"] / df["productivetreeha"],
    np.nan
)

df["fertilizer_per_productive_tree"] = np.where(
    df["productivetreeha"] > 0,
    df["totalfertha"] / df["productivetreeha"],
    np.nan
)

df["organic_fertilizer_per_productive_tree"] = np.where(
    df["productivetreeha"] > 0,
    df["totalorgfertha"] / df["productivetreeha"],
    np.nan
)

df["productive_tree_share"] = np.where(
    df["treedensityha"] > 0,
    df["productivetreeha"] / df["treedensityha"],
    np.nan
)

outcome = "yieldha"
district = "farmlocation"

full_features = [
    "totalwaterha",
    "water_per_productive_tree",
    "totalfertha",
    "fertilizer_per_productive_tree",
    "totalorgfertha",
    "organic_fertilizer_per_productive_tree",
    "pesticidesha",
    "treeageyears",
    "treedensityha",
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
    "n",
    "p",
    "k",
    "ec",
    "soilfertilityindex"
]

full_features = [
    x for x in full_features
    if x in df.columns and df[x].nunique(dropna=True) > 1
]

analysis = df[
    [outcome, district] + full_features
].copy()

for col in [outcome] + full_features:
    analysis[col] = pd.to_numeric(
        analysis[col],
        errors="coerce"
    )

analysis = analysis.dropna(
    subset=[outcome, district]
).reset_index(drop=True)

y = analysis[outcome].to_numpy()
groups = analysis[district].to_numpy()
eligible_districts = (
    analysis[district]
    .value_counts()
    .loc[lambda s: s >= 5]
    .index
    .tolist()
)

redundant_features = [
    "treedensityha",
    "n",
    "water_per_productive_tree",
    "totalfertha",
    "totalorgfertha",
    "fertilizer_per_productive_tree",
    "organic_fertilizer_per_productive_tree"
]

de_redundant_features = [
    x for x in full_features
    if x not in redundant_features
]

de_redundant_without_experience = [
    x for x in de_redundant_features
    if x != "experienceyears"
]

def make_model(feature_set):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median")
                        )
                    ]
                ),
                feature_set
            )
        ],
        remainder="drop"
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("xgb", XGBRegressor(**XGB_PARAMS))
        ]
    )

def evaluate_model(name, feature_set):
    X = analysis[feature_set]

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof_pred = np.full(len(analysis), np.nan)

    for train_idx, test_idx in kf.split(X):
        model = make_model(feature_set)
        model.fit(
            X.iloc[train_idx],
            y[train_idx]
        )
        oof_pred[test_idx] = model.predict(
            X.iloc[test_idx]
        )

    eligible_districts = (
        analysis[district]
        .value_counts()
        .loc[lambda s: s >= 5]
        .index
        .tolist()
    )

    lodo_pred = np.full(len(analysis), np.nan)

    for district_value in eligible_districts:
        test_idx = np.where(
            groups == district_value
        )[0]

        train_idx = np.where(
            groups != district_value
        )[0]

        model = make_model(feature_set)
        model.fit(
            X.iloc[train_idx],
            y[train_idx]
        )

        lodo_pred[test_idx] = model.predict(
            X.iloc[test_idx]
        )

    valid = ~np.isnan(lodo_pred)

    return {
        "specification": name,
        "predictors": len(feature_set),
        "features": "; ".join(feature_set),
        "fivefold_r2": r2_score(y, oof_pred),
        "fivefold_rmse_kg_ha": mean_squared_error(
            y,
            oof_pred
        ) ** 0.5,
        "fivefold_mae_kg_ha": mean_absolute_error(
            y,
            oof_pred
        ),
        "lodo_n": int(valid.sum()),
        "lodo_r2": r2_score(
            y[valid],
            lodo_pred[valid]
        ),
        "lodo_rmse_kg_ha": mean_squared_error(
            y[valid],
            lodo_pred[valid]
        ) ** 0.5,
        "lodo_mae_kg_ha": mean_absolute_error(
            y[valid],
            lodo_pred[valid]
        )
    }

specifications = [
    (
        "Full revised model",
        full_features
    ),
    (
        "Reduced model",
        de_redundant_features
    ),
    (
        "Reduced model minus experience",
        de_redundant_without_experience
    )
]

results = []

for name, features in specifications:
    print(
        f"Running: {name} "
        f"({len(features)} predictors)"
    )
    results.append(
        evaluate_model(
            name,
            features
        )
    )

results_df = pd.DataFrame(results)

full_row = results_df.loc[
    results_df["specification"]
    == "Full revised model"
].iloc[0]

results_df["fivefold_r2_drop_vs_full"] = (
    full_row["fivefold_r2"]
    - results_df["fivefold_r2"]
)

results_df["lodo_r2_drop_vs_full"] = (
    full_row["lodo_r2"]
    - results_df["lodo_r2"]
)

results_df.to_csv(
    OUTPUT_DIR / "deredundant_robustness_results.csv",
    index=False
)

feature_audit = pd.DataFrame({
    "full_leakage_free_features": pd.Series(
        full_features
    ),
    "removed_for_deredundancy": pd.Series(
        redundant_features
    ),
    "de_redundant_features": pd.Series(
        de_redundant_features
    )
})

feature_audit.to_csv(
    OUTPUT_DIR / "deredundant_feature_audit.csv",
    index=False
)

metadata = {
    "outcome": "yieldha",
    "outcome_units": "kg/ha",
    "n_observations": int(len(analysis)),
    "district_variable": district,
    "eligible_lodo_districts": [
        int(x) if isinstance(x, (int, np.integer))
        else str(x)
        for x in eligible_districts
    ],
    "removed_for_deredundancy": redundant_features,
    "retained_de_redundant_features": de_redundant_features,
    "xgb_parameters": XGB_PARAMS
}

with open(
    OUTPUT_DIR / "deredundant_model_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=4)

plot_data = results_df.copy()

plt.figure(figsize=(9, 5))
plt.bar(
    plot_data["specification"],
    plot_data["lodo_r2"],
    color=["#6c757d", "#2a6f97", "#90be6d"]
)

plt.ylim(
    min(0, plot_data["lodo_r2"].min() - 0.05),
    1.0
)

plt.ylabel("LODO R²")
plt.title("Robustness to predictor de-redundancy")
plt.xticks(
    rotation=12,
    ha="right"
)

for i, value in enumerate(
    plot_data["lodo_r2"]
):
    plt.text(
        i,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "deredundant_lodo_r2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n" + "=" * 85)
print("DE-REDUNDANT ROBUSTNESS RESULTS")
print("=" * 85)
print(
    results_df[
        [
            "specification",
            "predictors",
            "fivefold_r2",
            "fivefold_rmse_kg_ha",
            "lodo_r2",
            "lodo_rmse_kg_ha",
            "lodo_r2_drop_vs_full"
        ]
    ]
    .round(6)
    .to_string(index=False)
)

print("\nVariables removed for de-redundancy:")
for variable in redundant_features:
    print(f"- {variable}")

print("\nVariables retained in de-redundant model:")
for variable in de_redundant_features:
    print(f"- {variable}")

print("\nFiles saved:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(path.name)

print("\nCompleted.")