from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

OUTPUT_DIR = BASE_DIR / "results" / "output_ablation"
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

DISTRICT_NAMES = {
    1: "Khlung",
    2: "Mueang Chanthaburi",
    3: "Na Yai Am",
    4: "Pong Nam Ron",
    5: "Soi Dao",
    6: "Khao Khitchakut",
    7: "Laem Sing",
    8: "Tha Mai",
    9: "Makham",
    10: "Kaeng Hang Maeo"
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
    "watercostha",
    "fertilizercostha",
    "orgfertcostha",
    "labourwage",
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

analysis = df[[outcome, district] + full_features].copy()

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

def evaluate_specification(
    specification,
    feature_set,
    outcome_values,
    is_placebo=False
):
    X = analysis[feature_set]

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof = np.full(len(analysis), np.nan)

    for train_idx, test_idx in kf.split(X):
        model = make_model(feature_set)
        model.fit(X.iloc[train_idx], outcome_values[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])

    fivefold_r2 = r2_score(outcome_values, oof)
    fivefold_rmse = mean_squared_error(
        outcome_values,
        oof
    ) ** 0.5
    fivefold_mae = mean_absolute_error(
        outcome_values,
        oof
    )

    eligible = (
        analysis[district]
        .value_counts()
        .loc[lambda s: s >= 5]
        .index
        .tolist()
    )

    lodo_pred = np.full(len(analysis), np.nan)

    for district_value in eligible:
        test_idx = np.where(groups == district_value)[0]
        train_idx = np.where(groups != district_value)[0]

        model = make_model(feature_set)
        model.fit(X.iloc[train_idx], outcome_values[train_idx])
        lodo_pred[test_idx] = model.predict(X.iloc[test_idx])

    valid = ~np.isnan(lodo_pred)

    lodo_r2 = r2_score(
        outcome_values[valid],
        lodo_pred[valid]
    )
    lodo_rmse = mean_squared_error(
        outcome_values[valid],
        lodo_pred[valid]
    ) ** 0.5
    lodo_mae = mean_absolute_error(
        outcome_values[valid],
        lodo_pred[valid]
    )

    return {
        "specification": specification,
        "placebo": is_placebo,
        "predictors_retained": len(feature_set),
        "predictors_removed": (
            len(full_features) - len(feature_set)
        ),
        "removed_variables": (
            "; ".join(
                [x for x in full_features if x not in feature_set]
            )
            if len(feature_set) < len(full_features)
            else ""
        ),
        "fivefold_r2": fivefold_r2,
        "fivefold_rmse_kg_ha": fivefold_rmse,
        "fivefold_mae_kg_ha": fivefold_mae,
        "lodo_n": int(valid.sum()),
        "lodo_r2": lodo_r2,
        "lodo_rmse_kg_ha": lodo_rmse,
        "lodo_mae_kg_ha": lodo_mae
    }

management_inputs = [
    "totalwaterha",
    "water_per_productive_tree",
    "totalfertha",
    "fertilizer_per_productive_tree",
    "totalorgfertha",
    "organic_fertilizer_per_productive_tree",
    "pesticidesha",
    "ipm",
    "soilimprovement",
    "pdt",
    "creditaccess",
    "training",
    "extensionvisits",
    "gapscoreoverall"
]

orchard_structure = [
    "treeageyears",
    "treedensityha",
    "productivetreeha",
    "productive_tree_share",
    "plantmonthong",
    "plantkradumthong",
    "plantkanyao"
]

farmer_characteristics = [
    "experienceyears",
    "education",
    "age",
    "familymember",
    "groupmember"
]

soil_context = [
    "ph",
    "om",
    "n",
    "p",
    "k",
    "ec",
    "soilfertilityindex"
]

management_inputs = [
    x for x in management_inputs if x in full_features
]
orchard_structure = [
    x for x in orchard_structure if x in full_features
]
farmer_characteristics = [
    x for x in farmer_characteristics if x in full_features
]
soil_context = [
    x for x in soil_context if x in full_features
]

minimal_physical_inputs = [
    "totalwaterha",
    "totalfertha",
    "totalorgfertha",
    "pesticidesha"
]
minimal_physical_inputs = [
    x for x in minimal_physical_inputs
    if x in full_features
]

water_metrics = [
    "totalwaterha",
    "water_per_productive_tree"
]
water_metrics = [
    x for x in water_metrics
    if x in full_features
]

tree_structure = [
    "treeageyears",
    "treedensityha",
    "productivetreeha",
    "productive_tree_share"
]
tree_structure = [
    x for x in tree_structure
    if x in full_features
]

specifications = [
    (
        "Full revised model",
        full_features,
        y,
        False
    ),
    (
        "Minus water metrics",
        [
            x for x in full_features
            if x not in water_metrics
        ],
        y,
        False
    ),
    (
        "Minus orchard structure",
        [
            x for x in full_features
            if x not in orchard_structure
        ],
        y,
        False
    ),
    (
        "Minus physical input quantities",
        [
            x for x in full_features
            if x not in minimal_physical_inputs
        ],
        y,
        False
    ),
    (
        "Minus management and orchard block",
        [
            x for x in full_features
            if x not in (
                management_inputs + orchard_structure
            )
        ],
        y,
        False
    ),
    (
        "Minus tree age and experience",
        [
            x for x in full_features
            if x not in [
                "treeageyears",
                "experienceyears"
            ]
        ],
        y,
        False
    ),
    (
        "Placebo: permuted outcome",
        full_features,
        np.random.default_rng(RANDOM_STATE).permutation(y),
        True
    )
]

results = []

for name, feature_set, outcome_values, placebo in specifications:
    if len(feature_set) == 0:
        continue

    print(f"Running: {name}")
    result = evaluate_specification(
        specification=name,
        feature_set=feature_set,
        outcome_values=outcome_values,
        is_placebo=placebo
    )
    results.append(result)

results_df = pd.DataFrame(results)

full_lodo = results_df.loc[
    results_df["specification"] == "Full revised model",
    "lodo_r2"
].iloc[0]

full_fivefold = results_df.loc[
    results_df["specification"] == "Full revised model",
    "fivefold_r2"
].iloc[0]

results_df["fivefold_r2_drop_vs_full"] = (
    full_fivefold
    - results_df["fivefold_r2"]
)

results_df["lodo_r2_drop_vs_full"] = (
    full_lodo
    - results_df["lodo_r2"]
)

results_df = results_df[
    [
        "specification",
        "placebo",
        "predictors_retained",
        "predictors_removed",
        "removed_variables",
        "fivefold_r2",
        "fivefold_r2_drop_vs_full",
        "fivefold_rmse_kg_ha",
        "fivefold_mae_kg_ha",
        "lodo_n",
        "lodo_r2",
        "lodo_r2_drop_vs_full",
        "lodo_rmse_kg_ha",
        "lodo_mae_kg_ha"
    ]
]

results_df.to_csv(
    OUTPUT_DIR / "ablation_results.csv",
    index=False
)

block_definitions = {
    "management_inputs": management_inputs,
    "orchard_structure": orchard_structure,
    "farmer_characteristics": farmer_characteristics,
    "soil_context": soil_context
}

with open(
    OUTPUT_DIR / "predictor_blocks.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(block_definitions, f, indent=4)

plot_data = results_df.loc[
    ~results_df["placebo"]
].copy()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=plot_data,
    x="lodo_r2",
    y="specification",
    color="#2a6f97"
)
plt.xlim(
    min(-0.05, plot_data["lodo_r2"].min() - 0.05),
    1.0
)
plt.xlabel("Leave-one-district-out R²")
plt.ylabel("")
plt.title("Leakage-free feature-ablation results")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ablation_lodo_r2.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n" + "=" * 85)
print("LEAKAGE-FREE ABLATION RESULTS")
print("=" * 85)
print(
    results_df[
        [
            "specification",
            "predictors_retained",
            "fivefold_r2",
            "fivefold_rmse_kg_ha",
            "lodo_r2",
            "lodo_rmse_kg_ha"
        ]
    ]
    .round(6)
    .to_string(index=False)
)

print("\nPredictor blocks:")
for name, variables in block_definitions.items():
    print(f"\n{name} ({len(variables)} variables):")
    print(variables)

print("\nFiles saved:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(path.name)

print("\nCompleted.")