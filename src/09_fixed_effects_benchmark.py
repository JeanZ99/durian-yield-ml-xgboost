"""
09_fixed_effects_benchmark.py
──────────────────────────────
OLS district fixed-effects benchmark models (Table 5b).

Specs:
  M1 — Climate only
  M2 — Management only
  M3 — Full OLS + district FE

Outputs:
  - results/tables/table5b_benchmarks.csv
  - results/tables/table5b_ols_coefficients.csv

Run:
    python src/09_fixed_effects_benchmark.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import (DATA_PROC, TABLES_DIR, LOGS_DIR,
                    TARGET, GROUP_CV, CLIMATE_VARS, MANAGEMENT_VARS, FEATURES_ALL)
from utils import get_logger, ensure_dirs, save_csv

logger = get_logger("09_fixed_effects_benchmark", LOGS_DIR)

OLS_SPECS = {
    "M1_Climate_only":        CLIMATE_VARS,
    "M2_Management_only": [
        "Capital_ha", "Total_Fert_ha", "Labour_ha",
        "GAP_score_overall", "Tree_age_years", "Total_water_ha",
    ],
    "M3_Full_OLS_DistrictFE": FEATURES_ALL,
}


def main():
    ensure_dirs(TABLES_DIR, LOGS_DIR)
    df = pd.read_parquet(DATA_PROC / "durian_ml_processed.parquet")

    bench_rows, coef_dfs = [], []

    for spec_name, feat_list in OLS_SPECS.items():
        avail = [f for f in feat_list if f in df.columns]
        rhs = " + ".join(avail) + f" + C({GROUP_CV})"
        formula = f"{TARGET} ~ {rhs}"
        try:
            res = smf.ols(formula, data=df).fit()
            y_pred = res.fittedvalues.values
            y_true = df[TARGET].values
            bench_rows.append({
                "Specification": spec_name,
                "N_features":    len(avail),
                "R2":     round(res.rsquared, 4),
                "Adj_R2": round(res.rsquared_adj, 4),
                "RMSE":   round(np.sqrt(np.mean((y_true - y_pred)**2)), 2),
                "AIC":    round(res.aic, 1),
                "BIC":    round(res.bic, 1),
            })
            coef_df = pd.DataFrame({
                "Variable":    res.params.index,
                "Coefficient": res.params.values,
                "SE":          res.bse.values,
                "p_value":     res.pvalues.values,
                "Spec":        spec_name,
            })
            coef_dfs.append(coef_df)
            logger.info(f"{spec_name}: R2={res.rsquared:.4f}")
        except Exception as e:
            logger.error(f"{spec_name} failed: {e}")

    save_csv(pd.DataFrame(bench_rows), TABLES_DIR / "table5b_benchmarks.csv")
    if coef_dfs:
        save_csv(pd.concat(coef_dfs, ignore_index=True),
                 TABLES_DIR / "table5b_ols_coefficients.csv")
    logger.info("Benchmarks complete.")


if __name__ == "__main__":
    main()
