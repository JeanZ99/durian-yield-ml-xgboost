"""
07_tercile_analysis_water_intensity.py
───────────────────────────────────────
Robustness check: same tercile simulation as 06 but using
Water_intensity (Total_water_ha / Land) as the stratification variable.

Outputs:
  - results/tables/tableA1_water_intensity_tercile.csv

Run:
    python src/07_tercile_analysis_water_intensity.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (DATA_PROC, MODELS_DIR, TABLES_DIR, LOGS_DIR,
                    TARGET, FEATURES_ALL, TERCILE_VAR_ROBUST, TERCILE_LABELS)
from utils import get_logger, ensure_dirs, save_csv

logger = get_logger("07_tercile_water_intensity", LOGS_DIR)
TEMP_DELTAS = [0.0, 0.5, 1.0]


def bootstrap_ci(arr, n_boot=2000, ci=0.95):
    np.random.seed(42)
    means = [np.random.choice(arr, len(arr), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def main():
    ensure_dirs(TABLES_DIR, LOGS_DIR)
    df = pd.read_parquet(DATA_PROC / "durian_ml_processed.parquet")
    feats = [f for f in FEATURES_ALL if f in df.columns]

    if TERCILE_VAR_ROBUST not in df.columns:
        logger.error(f"{TERCILE_VAR_ROBUST} not found — run 01_data_prep.py first.")
        return

    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / "xgb_final.json"))

    df["Tercile_WI"] = pd.qcut(df[TERCILE_VAR_ROBUST], 3, labels=TERCILE_LABELS)

    rows = []
    for label in TERCILE_LABELS:
        sub = df[df["Tercile_WI"] == label]
        X_base = sub[feats].copy()
        y_base = model.predict(X_base)

        for delta in TEMP_DELTAS:
            X_warm = X_base.copy()
            X_warm["Temp_annual"] = X_warm["Temp_annual"] + delta
            if "Temp_grow" in feats:
                X_warm["Temp_grow"] = X_warm["Temp_grow"] + delta

            y_pred = model.predict(X_warm)
            dy = y_pred - y_base
            mean_dy = dy.mean()
            lo, hi = bootstrap_ci(dy)

            rows.append({
                "Irrigation_Tercile_WI": label,
                "N":                     len(sub),
                "Baseline_Yield_mean":   round(y_base.mean(), 1),
                "Temp_Delta_C":          delta,
                "Predicted_dY_kg_ha":    round(mean_dy, 1),
                "CI95_lower":            round(lo, 1),
                "CI95_upper":            round(hi, 1),
            })

    save_csv(pd.DataFrame(rows), TABLES_DIR / "tableA1_water_intensity_tercile.csv")
    logger.info("Appendix A1 (Water_intensity tercile) saved.")


if __name__ == "__main__":
    main()
