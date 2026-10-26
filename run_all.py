#!/usr/bin/env python3
"""
Master Reproduction Script for:
"Predicting yield variation among GAP-certified durian farms in Chanthaburi, Thailand:
The roles of orchard structure and management"
Authors: Jintawat Phaiboontrirat, Takuji W. Tsusaka, Sushil K. Himanshu, Malay K. Pramanik

Reproduces all tables (Tables 1-8) and figures (Figures 2-4) in the manuscript
using the audited dataset (n = 393 farms) and fixed random seeds (random_state = 42).
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("1. Pre-modeling audit & exclusion diagnostics (Table 3)", "reanalysis_durian.py"),
    ("2. Full revised 33-predictor model: 5-fold CV & LODO validation (Tables 2 & 4, Fig. 2)", "run_leakage_free_model.py"),
    ("3. Feature-block ablation & outcome permutation placebo (Table 5, Fig. 3)", "run_ablation_and_robustness.py"),
    ("4. Collinearity & redundancy diagnostics (Section 3.5)", "run_redundancy_diagnostics.py"),
    ("5. Reduced 26-predictor model & experience sensitivity (Table 6)", "run_deredundant_robustness.py"),
    ("6. Full-sample TreeSHAP attributions for the 26-predictor reduced model (Tables 7 & 8, Fig. 4)", "run_deredundant_shap_ale.py"),
]

def main():
    print("=" * 80)
    print("RUNNING FULL MANUSCRIPT REPRODUCTION PIPELINE (V.14)")
    print("=" * 80)
    for title, script_name in SCRIPTS:
        print(f"\n>>> {title} [{script_name}]")
        script_path = BASE_DIR / script_name
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), check=True)

    # Consolidate key manuscript tables into results/tables/
    tables_dir = BASE_DIR / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        BASE_DIR / "results" / "output_revised_model" / "analysis_descriptive_statistics.csv",
        tables_dir / "Table2_descriptive_statistics.csv"
    )
    shutil.copy2(
        BASE_DIR / "results" / "output_revised_model" / "lodo_results_by_district.csv",
        tables_dir / "Table4_lodo_by_district.csv"
    )
    shutil.copy2(
        BASE_DIR / "results" / "output_ablation" / "ablation_results.csv",
        tables_dir / "Table5_feature_ablation_and_placebo.csv"
    )
    shutil.copy2(
        BASE_DIR / "results" / "output_reduced_model" / "deredundant_robustness_results.csv",
        tables_dir / "Table6_reduced_model_robustness.csv"
    )
    shutil.copy2(
        BASE_DIR / "results" / "output_reduced_shap" / "deredundant_shap_feature_importance.csv",
        tables_dir / "Table7_reduced_model_shap_features.csv"
    )
    shutil.copy2(
        BASE_DIR / "results" / "output_reduced_shap" / "deredundant_shap_block_importance.csv",
        tables_dir / "Table8_reduced_model_shap_blocks.csv"
    )

    print("\n" + "=" * 80)
    print("ALL MANUSCRIPT TABLES (TABLES 2, 4, 5, 6, 7, 8) REPRODUCED SUCCESSFULLY!")
    print(f"Consolidated tables saved in: {tables_dir}")
    print("=" * 80)

if __name__ == "__main__":
    import shutil
    main()
