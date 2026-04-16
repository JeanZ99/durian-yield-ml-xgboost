"""
04_shap_analysis.py
SHAP-based interpretation: importance table, beeswarm, bar, PDP, interaction.
Run: python src/04_shap_analysis.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (DATA_PROC, MODELS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR,
                    TARGET, FEATURES_ALL,
                    IRRIGATION_VARS, MANAGEMENT_VARS, AGE_EXPERIENCE_VARS,
                    CLIMATE_VARS, SOIL_VARS, FARM_STRUCT_VARS, SOCIOECONOMIC_VARS)
from utils import get_logger, ensure_dirs, save_csv

logger = get_logger("04_shap_analysis", LOGS_DIR)

BLOCK_MAP = {}
for v in IRRIGATION_VARS:     BLOCK_MAP[v] = "Irrigation"
for v in MANAGEMENT_VARS:     BLOCK_MAP[v] = "Management"
for v in AGE_EXPERIENCE_VARS: BLOCK_MAP[v] = "Age & Experience"
for v in CLIMATE_VARS:        BLOCK_MAP[v] = "Climate"
for v in SOIL_VARS:           BLOCK_MAP[v] = "Soil"
for v in FARM_STRUCT_VARS:    BLOCK_MAP[v] = "Farm Structure"
for v in SOCIOECONOMIC_VARS:  BLOCK_MAP[v] = "Socioeconomic"


def main():
    ensure_dirs(TABLES_DIR, FIGURES_DIR, LOGS_DIR)
    df = pd.read_parquet(DATA_PROC / "durian_ml_processed.parquet")
    feats = [f for f in FEATURES_ALL if f in df.columns]
    X = df[feats]

    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / "xgb_final.json"))
    logger.info("Computing SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Table 6
    imp_df = pd.DataFrame({
        "Feature":      feats,
        "Mean_AbsSHAP": np.abs(shap_values).mean(axis=0),
        "Block":        [BLOCK_MAP.get(f, "Other") for f in feats],
    }).sort_values("Mean_AbsSHAP", ascending=False).reset_index(drop=True)
    imp_df["Rank"] = range(1, len(imp_df) + 1)
    save_csv(imp_df, TABLES_DIR / "table6_shap_importance.csv")

    # Top-20 column indices
    top20_feats = imp_df["Feature"].tolist()[:20]
    idx_map = {f: i for i, f in enumerate(feats)}
    cols = [idx_map[f] for f in top20_feats if f in idx_map]

    shap.summary_plot(shap_values[:, cols], X.iloc[:, cols],
                      feature_names=top20_feats, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values[:, cols], X.iloc[:, cols],
                      feature_names=top20_feats, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar_top20.png", dpi=300, bbox_inches="tight")
    plt.close()

    if "Total_water_ha" in feats:
        shap.dependence_plot("Total_water_ha", shap_values, X,
                             interaction_index=None, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_pdp_total_water.png", dpi=300, bbox_inches="tight")
        plt.close()

    if "Total_water_ha" in feats and "Temp_annual" in feats:
        shap.dependence_plot("Total_water_ha", shap_values, X,
                             interaction_index="Temp_annual", show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_interaction_water_temp.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info("SHAP analysis complete.")


if __name__ == "__main__":
    main()
