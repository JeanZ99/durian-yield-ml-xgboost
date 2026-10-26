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
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx"
    if (BASE_DIR / "data" / "Cleaned_Durian_Data_ML.xlsx").exists()
    else BASE_DIR / "Cleaned_Durian_Data_ML.xlsx"
)

OUTPUT_DIR = BASE_DIR / "results" / "output_revised_model"
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
    "gaps coreoverall".replace(" ", ""),
    "gaps core overall".replace(" ", ""),
    "gapscoreoverall",
    "land",
    "capitalha",
    "labor",
    "familylabor",
    "hiredlabor",
    "migrantworkers"
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

df["district_name"] = df["farmlocation"].map(DISTRICT_NAMES)

outcome = "yieldha"
district = "farmlocation"

forbidden_predictors = {
    "id",
    "year",
    "yieldha",
    "totalyield",
    "totalyieldsum",
    "price",
    "totalincome",
    "totalincomeha",
    "revenueha",
    "waterintensity",
    "fertintensity",
    "cpha",
    "cpeha",
    "meha",
    "chha",
    "cfha",
    "clha",
    "etcha",
    "waterprice",
    "fertilizerprice",
    "orgfertprice",
    "watercostha",
    "fertilizercostha",
    "orgfertcostha",
    "capitalha",
    "labor",
    "familylabor",
    "hiredlabor",
    "migrant",
    "migrantworkers",
    "tempgrow",
    "tempannual",
    "raingrow",
    "rainannual",
    "farmlocation",
    "district_name"
}

candidate_predictors = [
    "totalwaterha",
    "water_per_productive_tree",
    "totalfertha",
    "fertilizer_per_productive_tree",
    "totalorgfertha",
    "organic_fertilizer_per_productive_tree",
    "pesticidesha",
    "primaryirrigation",
    "modernirrigation",
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

predictors = [
    col for col in candidate_predictors
    if col in df.columns and col not in forbidden_predictors
]

required_columns = [outcome, district] + predictors
analysis = df[required_columns].copy()
analysis = analysis.dropna(subset=[outcome, district]).reset_index(drop=True)

for col in predictors:
    analysis[col] = pd.to_numeric(analysis[col], errors="coerce")

constant_predictors = [
    col for col in predictors
    if analysis[col].nunique(dropna=True) <= 1
]

predictors = [
    col for col in predictors
    if col not in constant_predictors
]

analysis = analysis[[outcome, district] + predictors].copy()

print("=" * 75)
print("LEAKAGE-FREE DURAN YIELD MODEL")
print("=" * 75)
print(f"Observations: {len(analysis)}")
print(f"Predictors: {len(predictors)}")
print(f"Outcome: {outcome} (kg/ha)")
print(f"Districts: {analysis[district].nunique()}")
print("\nRemoved by design:")
print(sorted(forbidden_predictors))
print("\nConstant predictors removed:")
print(constant_predictors)
print("\nFinal predictors:")
print(predictors)

X = analysis[predictors]
y = analysis[outcome].to_numpy()
groups = analysis[district].to_numpy()

numeric_features = predictors

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            ),
            numeric_features
        )
    ],
    remainder="drop"
)

def make_model():
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("xgb", XGBRegressor(**XGB_PARAMS))
        ]
    )

def metrics_frame(y_true, y_pred, label):
    return pd.DataFrame(
        [{
            "specification": label,
            "n": len(y_true),
            "r2": r2_score(y_true, y_pred),
            "rmse_kg_ha": mean_squared_error(y_true, y_pred) ** 0.5,
            "mae_kg_ha": mean_absolute_error(y_true, y_pred)
        }]
    )

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_predictions = np.full(len(analysis), np.nan)
fold_rows = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    model = make_model()
    model.fit(X.iloc[train_idx], y[train_idx])
    pred = model.predict(X.iloc[test_idx])

    oof_predictions[test_idx] = pred

    fold_rows.append({
        "fold": fold,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "r2": r2_score(y[test_idx], pred),
        "rmse_kg_ha": mean_squared_error(y[test_idx], pred) ** 0.5,
        "mae_kg_ha": mean_absolute_error(y[test_idx], pred)
    })

fold_results = pd.DataFrame(fold_rows)
fivefold_overall = metrics_frame(
    y,
    oof_predictions,
    "Full revised XGBoost (33 predictors), five-fold CV"
)

district_counts = analysis[district].value_counts().sort_index()
eligible_districts = district_counts[
    district_counts >= 5
].index.tolist()

lodo_predictions = np.full(len(analysis), np.nan)
lodo_rows = []

for district_value in eligible_districts:
    test_idx = np.where(groups == district_value)[0]
    train_idx = np.where(groups != district_value)[0]

    model = make_model()
    model.fit(X.iloc[train_idx], y[train_idx])
    pred = model.predict(X.iloc[test_idx])

    lodo_predictions[test_idx] = pred

    lodo_rows.append({
        "district_code": district_value,
        "district_name": DISTRICT_NAMES.get(
            district_value,
            str(district_value)
        ),
        "n_test": len(test_idx),
        "r2": r2_score(y[test_idx], pred),
        "rmse_kg_ha": mean_squared_error(y[test_idx], pred) ** 0.5,
        "mae_kg_ha": mean_absolute_error(y[test_idx], pred)
    })

lodo_results = pd.DataFrame(lodo_rows)
valid_lodo = ~np.isnan(lodo_predictions)

lodo_overall = metrics_frame(
    y[valid_lodo],
    lodo_predictions[valid_lodo],
    "Full revised XGBoost (33 predictors), LODO"
)

full_model = make_model()
full_model.fit(X, y)

xgb_model = full_model.named_steps["xgb"]
importance = pd.DataFrame({
    "feature": predictors,
    "gain_importance": xgb_model.feature_importances_
}).sort_values("gain_importance", ascending=False)

importance["gain_importance_percent"] = (
    100
    * importance["gain_importance"]
    / importance["gain_importance"].sum()
)

full_summary = (
    analysis[predictors + [outcome]]
    .describe()
    .T
    .reset_index()
    .rename(columns={"index": "variable"})
)

oof_output = analysis[[district, outcome]].copy()
oof_output["district_name"] = oof_output[district].map(DISTRICT_NAMES)
oof_output["predicted_yield_5fold_oof"] = oof_predictions
oof_output["residual_5fold_oof"] = (
    oof_output[outcome]
    - oof_output["predicted_yield_5fold_oof"]
)

lodo_output = analysis[[district, outcome]].copy()
lodo_output["district_name"] = lodo_output[district].map(DISTRICT_NAMES)
lodo_output["predicted_yield_lodo"] = lodo_predictions
lodo_output["residual_lodo"] = (
    lodo_output[outcome]
    - lodo_output["predicted_yield_lodo"]
)

model_metadata = {
    "outcome": outcome,
    "outcome_unit": "kg/ha",
    "n_observations": int(len(analysis)),
    "n_predictors": int(len(predictors)),
    "predictors": predictors,
    "forbidden_predictors": sorted(forbidden_predictors),
    "constant_predictors_removed": constant_predictors,
    "fivefold_r2": float(fivefold_overall.loc[0, "r2"]),
    "fivefold_rmse_kg_ha": float(
        fivefold_overall.loc[0, "rmse_kg_ha"]
    ),
    "fivefold_mae_kg_ha": float(
        fivefold_overall.loc[0, "mae_kg_ha"]
    ),
    "lodo_r2": float(lodo_overall.loc[0, "r2"]),
    "lodo_rmse_kg_ha": float(
        lodo_overall.loc[0, "rmse_kg_ha"]
    ),
    "lodo_mae_kg_ha": float(
        lodo_overall.loc[0, "mae_kg_ha"]
    ),
    "lodo_eligible_districts": [
        int(x) for x in eligible_districts
    ],
    "lodo_excluded_districts_under_n5": [
        int(x) for x in district_counts[
            district_counts < 5
        ].index.tolist()
    ],
    "xgboost_parameters": XGB_PARAMS
}

fold_results.to_csv(
    OUTPUT_DIR / "fivefold_results.csv",
    index=False
)
fivefold_overall.to_csv(
    OUTPUT_DIR / "fivefold_overall_metrics.csv",
    index=False
)
lodo_results.to_csv(
    OUTPUT_DIR / "lodo_results_by_district.csv",
    index=False
)
lodo_overall.to_csv(
    OUTPUT_DIR / "lodo_overall_metrics.csv",
    index=False
)
importance.to_csv(
    OUTPUT_DIR / "feature_importance_gain.csv",
    index=False
)
full_summary.to_csv(
    OUTPUT_DIR / "analysis_descriptive_statistics.csv",
    index=False
)
oof_output.to_csv(
    OUTPUT_DIR / "fivefold_oof_predictions.csv",
    index=False
)
lodo_output.to_csv(
    OUTPUT_DIR / "lodo_predictions.csv",
    index=False
)

with open(
    OUTPUT_DIR / "model_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(model_metadata, f, indent=4)

plt.figure(figsize=(10, 8))
top = importance.head(20).sort_values("gain_importance")
sns.barplot(
    data=top,
    x="gain_importance",
    y="feature",
    color="#2a6f97"
)
plt.title("Leakage-free XGBoost feature importance")
plt.xlabel("XGBoost gain importance")
plt.ylabel("")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "feature_importance_gain_top20.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(7, 7))
sns.scatterplot(
    x=y,
    y=oof_predictions,
    color="#2a6f97",
    s=45
)
low = min(np.min(y), np.min(oof_predictions))
high = max(np.max(y), np.max(oof_predictions))
plt.plot([low, high], [low, high], "--", color="black")
plt.xlabel("Observed yield (kg/ha)")
plt.ylabel("Five-fold OOF predicted yield (kg/ha)")
plt.title("Leakage-free XGBoost: observed versus OOF predictions")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "fivefold_oof_observed_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n" + "=" * 75)
print("RESULTS")
print("=" * 75)
print("\nFive-fold cross-validation:")
print(fivefold_overall.round(6).to_string(index=False))

print("\nLODO validation:")
print(lodo_overall.round(6).to_string(index=False))

print("\nLODO by district:")
print(lodo_results.round(6).to_string(index=False))

print("\nTop 20 XGBoost gain-importance features:")
print(
    importance.head(20)
    .round(6)
    .to_string(index=False)
)

print("\nSaved outputs:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(path.name)

print("\nCompleted.")