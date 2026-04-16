"""
03_validation_5fold_lodo.py
Leakage-free 5-fold CV and Leave-One-District-Out (LODO) validation.
Run: python src/03_validation_5fold_lodo.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, LeaveOneGroupOut

from config import (DATA_PROC, MODELS_DIR, TABLES_DIR, LOGS_DIR,
                    TARGET, GROUP_CV, CV_FOLDS, RANDOM_SEED, FEATURES_ALL)
from utils import get_logger, ensure_dirs, regression_metrics, save_csv, print_metrics

logger = get_logger("03_validation", LOGS_DIR)


def main():
    ensure_dirs(TABLES_DIR, LOGS_DIR)
    df = pd.read_parquet(DATA_PROC / "durian_ml_processed.parquet")
    feats = [f for f in FEATURES_ALL if f in df.columns]
    X = df[feats].values
    y = df[TARGET].values
    groups = df[GROUP_CV].values

    with open(MODELS_DIR / "xgb_params.json") as f:
        params = json.load(f)
    logger.info(f"N={len(y)} | Features={len(feats)}")

    # 5-fold CV
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_rows = []
    for fold, (tr, val) in enumerate(kf.split(X), 1):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr])
        met = regression_metrics(y[val], m.predict(X[val]))
        met["Fold"] = fold
        print_metrics(f"Fold {fold}", met)
        cv_rows.append(met)
    df_cv = pd.DataFrame(cv_rows)
    save_csv(df_cv, TABLES_DIR / "table5_cv_metrics.csv")

    # LODO
    logo = LeaveOneGroupOut()
    lodo_rows = []
    for tr, val in logo.split(X, y, groups):
        dist = groups[val[0]]
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr])
        met = regression_metrics(y[val], m.predict(X[val]))
        met["District"] = dist; met["N_val"] = len(val)
        print_metrics(f"LODO {dist}", met)
        lodo_rows.append(met)
    df_lodo = pd.DataFrame(lodo_rows)
    save_csv(df_lodo, TABLES_DIR / "table5_lodo_metrics.csv")

    # Combined Table 5
    combined = pd.DataFrame({
        "Metric": ["R2", "RMSE (kg/ha)", "MAE (kg/ha)"],
        "5Fold_Mean": df_cv[["R2","RMSE","MAE"]].mean().round(4).values,
        "5Fold_SD":   df_cv[["R2","RMSE","MAE"]].std().round(4).values,
        "LODO_Mean":  df_lodo[["R2","RMSE","MAE"]].mean().round(4).values,
        "LODO_SD":    df_lodo[["R2","RMSE","MAE"]].std().round(4).values,
    })
    save_csv(combined, TABLES_DIR / "table5_combined_performance.csv")
    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
