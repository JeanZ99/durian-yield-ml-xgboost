"""
10_make_tables_figures.py
─────────────────────────
Collect all CSV/figure outputs and copy to manuscript/tables_for_submission/
and manuscript/figures_for_submission/ for journal submission.

Run after all scripts 01-09 have completed:
    python src/10_make_tables_figures.py
"""

import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")

from config import TABLES_DIR, FIGURES_DIR, ROOT, LOGS_DIR
from utils import get_logger, ensure_dirs

logger = get_logger("10_make_tables_figures", LOGS_DIR)

SUBMIT_TABLES = ROOT / "manuscript" / "tables_for_submission"
SUBMIT_FIGS   = ROOT / "manuscript" / "figures_for_submission"

TABLE_FILES = {
    "Table5_Performance":  "table5_combined_performance.csv",
    "Table5b_Benchmarks":  "table5b_benchmarks.csv",
    "Table5c_Ablation":    "table5c_ablation.csv",
    "Table6_SHAP":         "table6_shap_importance.csv",
    "Table7_Scenarios":    "table7_scenario_climate_warming.csv",
    "Table8_Tercile":      "table8_tercile_total_water.csv",
    "TableA1_Robustness":  "tableA1_water_intensity_tercile.csv",
    "TableB1_Balance":     "tableB1_balance_tests.csv",
}

FIGURE_FILES = {
    "Figure2_SHAP_Beeswarm": "shap_beeswarm.png",
    "Figure3_SHAP_Bar":      "shap_bar_top20.png",
    "Figure4_PDP_Water":     "shap_pdp_total_water.png",
    "Figure5_Interaction":   "shap_interaction_water_temp.png",
}


def main():
    ensure_dirs(SUBMIT_TABLES, SUBMIT_FIGS)

    for out_name, fname in TABLE_FILES.items():
        src = TABLES_DIR / fname
        if not src.exists():
            logger.warning(f"Missing: {fname}")
            continue
        dst = SUBMIT_TABLES / f"{out_name}.csv"
        shutil.copy2(src, dst)
        logger.info(f"Copied {out_name}")

    for out_name, fname in FIGURE_FILES.items():
        src = FIGURES_DIR / fname
        if not src.exists():
            logger.warning(f"Missing: {fname}")
            continue
        try:
            from PIL import Image
            img = Image.open(src)
            tiff = SUBMIT_FIGS / f"{out_name}.tiff"
            img.save(tiff, dpi=(300, 300), compression="tiff_lzw")
            logger.info(f"Converted {out_name} -> TIFF")
        except ImportError:
            shutil.copy2(src, SUBMIT_FIGS / f"{out_name}.png")
            logger.info(f"Copied {out_name} (no Pillow — PNG)")

    logger.info("All submission assets assembled.")


if __name__ == "__main__":
    main()
