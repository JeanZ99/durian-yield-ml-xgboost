"""
config.py — Central configuration for all pipeline scripts.
"""

from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT / "data" / "raw"
DATA_INTERIM  = ROOT / "data" / "interim"
DATA_PROC     = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"
TABLES_DIR    = RESULTS_DIR / "tables"
FIGURES_DIR   = RESULTS_DIR / "figures"
MODELS_DIR    = RESULTS_DIR / "models"
LOGS_DIR      = RESULTS_DIR / "logs"

RAW_DATA_FILE = DATA_RAW / "Cleaned_Durian_Data_ML.xlsx"
DATA_DICT     = DATA_RAW / "Data-Dictionary-ML.xlsx"

TARGET   = "Yield_ha"
GROUP_CV = "Farm_location"

IRRIGATION_VARS = [
    "Total_water_ha",
    "Water_set1", "Water_set2", "Water_set3",
    "Water_set4", "Water_set5", "Water_set6",
    "Water_intensity", "Modern_Irrigation", "Primary_Irrigation",
]

MANAGEMENT_VARS = [
    "Capital_ha", "Total_Fert_ha", "Total_OrgFert_ha",
    "Fertilizer_Cost_ha", "OrgFert_Cost_ha", "Pesticides_ha",
    "Labour_ha", "Tree_Density_ha", "Productive_tree_ha", "Pdt",
    "Soil_improvement", "GAP_score_overall",
    "Rk", "Sup", "Fm", "Wm", "Phm", "Wsm", "IPM", "Fst", "Trs", "Bz",
]

AGE_EXPERIENCE_VARS = [
    "Tree_age_years", "Experience_years", "Age", "Education",
    "Training", "Extension_visits", "Group_member",
    "Knowledge_GAP", "Credit_Access",
]

CLIMATE_VARS = ["Temp_annual", "Rain_annual", "Temp_grow", "Rain_grow"]

SOIL_VARS = ["PH", "OM", "N", "P", "K", "EC", "Soil_fertility_index"]

FARM_STRUCT_VARS = [
    "Land", "Tree_Density_ha", "Productive_tree_ha",
    "Plant_Monthong", "Plant_Chani", "Plant_Kradumthong",
    "Plant_Kanyao", "Plant_Other", "Lt",
]

SOCIOECONOMIC_VARS = [
    "Family_Member", "Family_labor", "Hired_labor",
    "Export_chain", "Du", "Oc", "Ls", "Ofw",
]

FEATURES_ALL = list(dict.fromkeys(
    IRRIGATION_VARS + MANAGEMENT_VARS + AGE_EXPERIENCE_VARS
    + CLIMATE_VARS + SOIL_VARS + FARM_STRUCT_VARS + SOCIOECONOMIC_VARS
))

ABLATION_SPECS = {
    "Full model":              FEATURES_ALL,
    "Minus irrigation block":  [f for f in FEATURES_ALL if f not in IRRIGATION_VARS],
    "Minus management block":  [f for f in FEATURES_ALL if f not in MANAGEMENT_VARS],
    "Minus age & experience":  [f for f in FEATURES_ALL if f not in AGE_EXPERIENCE_VARS],
    "Climate + soil only":     CLIMATE_VARS + SOIL_VARS,
}

XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "n_jobs": -1, "tree_method": "hist",
}

CV_FOLDS    = 5
RANDOM_SEED = 42

SCENARIO_TEMP_DELTA = [0, +0.5, +1.0, +1.5, +2.0]
SCENARIO_RAIN_DELTA = [0, -10, -20, +10, +20]

TERCILE_VAR_MAIN   = "Total_water_ha"
TERCILE_VAR_ROBUST = "Water_intensity"
TERCILE_LABELS     = ["Low (T1)", "Medium (T2)", "High (T3)"]
