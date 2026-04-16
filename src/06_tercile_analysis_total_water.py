"""
06_tercile_analysis_total_water.py
───────────────────────────────────
Main tercile analysis: stratify farms into Low/Medium/High irrigation groups
based on Total_water_ha, then simulate +0.5 and +1.0 degC warming within each group.

Outputs:
  - results/tables/table8_tercile_total_water.csv
  - results/tables/tableB1_balance_tests.csv

Run:
    python src/06_tercile_analysis_total_water.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from config import (DATA_PROC, MODELS_DIR, TABLES_DIR, LOGS_DIR,
                    TARGET, FEATURES_ALL, TERCILE_VAR_MAIN, TERCILE_LABELS)
from utils import get_logger, ensure_dirs, save_csv

logger = get_logger("06_tercile_total_water", LOGS_DIR)

TEMP_DELTAS = [0.0, 0.5, 1.0]
COVARIATE_BALANCE = [
    "Land", "Capital_ha", "Soil_fertility_index",
    "Credit_Access", "Experience_years", "GAP_score_overall",
]


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

    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / "xgb_final.json"))

    df["Tercile"] = pd.qcut(df[TERCILE_VAR_MAIN], 3, labels=TERCILE_LABELS)

    rows = []
    for label in TERCILE_LABELS:
        sub = df[df["Tercile"] == label]
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
                "Irrigation_Tercile":  label,
                "N":                   len(sub),
                "Baseline_Yield_mean": round(y_base.mean(), 1),
                "Temp_Delta_C":        delta,
                "Predicted_dY_kg_ha":  round(mean_dy, 1),
                "CI95_lower":          round(lo, 1),
                "CI95_upper":          round(hi, 1),
            })

    save_csv(pd.DataFrame(rows), TABLES_DIR / "table8_tercile_total_water.csv")
    logger.info("Table 8 saved.")

    b_rows = []
    for var in COVARIATE_BALANCE:
        if var not in df.columns:
            continue
        groups = [df[df["Tercile"] == t][var].dropna().values for t in TERCILE_LABELS]
        h_stat, p_val = stats.kruskal(*groups)
        means = {t: round(df[df["Tercile"] == t][var].mean(), 3) for t in TERCILE_LABELS}
        b_rows.append({
            "Variable":     var,
            "Mean_T1_Low":  means[TERCILE_LABELS[0]],
            "Mean_T2_Med":  means[TERCILE_LABELS[1]],
            "Mean_T3_High": means[TERCILE_LABELS[2]],
            "KW_H":         round(h_stat, 3),
            "p_value":      round(p_val, 4),
            "Balanced":     "Yes" if p_val > 0.05 else "No",
        })

    save_csv(pd.DataFrame(b_rows), TABLES_DIR / "tableB1_balance_tests.csv")
    logger.info("Balance tests (Appendix B) saved.")


if __name__ == "__main__":
    main()
