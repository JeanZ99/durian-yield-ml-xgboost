"""
08_feature_ablation.py
──────────────────────
Feature ablation: evaluate XGBoost on progressively reduced feature sets
using 5-fold CV and LODO to quantify each block's contribution.

Ablation specs (from config.ABLATION_SPECS):
  1. Full model
  2. Minus irrigation block
  3. Minus management block
  4. Minus age & experience block
  5. Climate + soil only

Outputs:
  - results/tables/table5c_ablation.csv

Run:
    python src/08_feature_ablation.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, LeaveOneGroupOut

from config import (DATA_PROC, MODELS_DIR, TABLES_DIR, LOGS_DIR,
                    TARGET, GROUP_CV, CV_FOLDS, RANDOM_SEED, ABLATION_SPECS)
from utils import get_logger, ensure_dirs, regression_metrics, save_csv

logger = get_logger("08_feature_ablation", LOGS_DIR)


def evaluate_spec(spec_name, feat_list, df, params):
    feats = [f for f in feat_list if f in df.columns]
    if not feats:
        return None

    X = df[feats].values
    y = df[TARGET].values
    groups = df[GROUP_CV].values

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_r2 = [regression_metrics(y[val], xgb.XGBRegressor(**params).fit(X[tr], y[tr]).predict(X[val]))["R2"]
             for tr, val in kf.split(X)]

    logo = LeaveOneGroupOut()
    lodo_r2 = [regression_metrics(y[val], xgb.XGBRegressor(**params).fit(X[tr], y[tr]).predict(X[val]))["R2"]
               for tr, val in logo.split(X, y, groups)]

    return {
        "Specification":  spec_name,
        "N_features":     len(feats),
        "5Fold_R2_mean":  round(np.mean(cv_r2), 4),
        "5Fold_R2_SD":    round(np.std(cv_r2), 4),
        "LODO_R2_mean":   round(np.mean(lodo_r2), 4),
        "LODO_R2_SD":     round(np.std(lodo_r2), 4),
    }


def main():
    ensure_dirs(TABLES_DIR, LOGS_DIR)
    df = pd.read_parquet(DATA_PROC / "durian_ml_processed.parquet")

    with open(MODELS_DIR / "xgb_params.json") as f:
        params = json.load(f)

    rows = []
    for name, feat_list in ABLATION_SPECS.items():
        logger.info(f"Spec: {name}")
        result = evaluate_spec(name, feat_list, df, params)
        if result:
            rows.append(result)
            logger.info(f"  5CV R2={result['5Fold_R2_mean']:.4f}  LODO R2={result['LODO_R2_mean']:.4f}")

    df_abl = pd.DataFrame(rows)
    full_cv   = df_abl.loc[df_abl["Specification"] == "Full model", "5Fold_R2_mean"].values[0]
    full_lodo = df_abl.loc[df_abl["Specification"] == "Full model", "LODO_R2_mean"].values[0]
    df_abl["Delta_5Fold_R2"] = (df_abl["5Fold_R2_mean"] - full_cv).round(4)
    df_abl["Delta_LODO_R2"]  = (df_abl["LODO_R2_mean"]  - full_lodo).round(4)

    save_csv(df_abl, TABLES_DIR / "table5c_ablation.csv")
    logger.info("Feature ablation complete.")


if __name__ == "__main__":
    main()
