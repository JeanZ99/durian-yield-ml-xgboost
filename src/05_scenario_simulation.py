"""
05_scenario_simulation.py
Counterfactual scenario simulations: water step-up, fertilizer step-up,
climate warming x irrigation tercile.
Run: python src/05_scenario_simulation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (DATA_PROC, MODELS_DIR, TABLES_DIR, LOGS_DIR,
                    TARGET, FEATURES_ALL, SCENARIO_TEMP_DELTA, TERCILE_VAR_MAIN)
from utils import get_logger, ensure_dirs, save_csv

logger = get_logger("05_scenario_simulation", LOGS_DIR)


def build_baseline(df, feats):
    base = {}
    for f in feats:
        col = df[f]
        if col.dtype == object or col.nunique() <= 5:
            base[f] = col.mode()[0]
        else:
            base[f] = col.median()
    return pd.Series(base)


def main():
    ensure_dirs(TABLES_DIR, LOGS_DIR)
    df = pd.read_parquet(DATA_PROC / "durian_ml_processed.parquet")
    feats = [f for f in FEATURES_ALL if f in df.columns]

    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / "xgb_final.json"))
    baseline = build_baseline(df, feats)

    # Scenario A: Water step-up
    if "Total_water_ha" in feats:
        vals = np.linspace(df["Total_water_ha"].quantile(0.10),
                           df["Total_water_ha"].quantile(0.90), 15)
        rows = []
        for v in vals:
            r = baseline.copy(); r["Total_water_ha"] = v
            pred = model.predict(r[feats].values.reshape(1, -1))[0]
            rows.append({"Total_water_ha": v, "Predicted_Yield_ha": round(pred, 2)})
        save_csv(pd.DataFrame(rows), TABLES_DIR / "table7_scenario_water.csv")

    # Scenario B: Fertilizer step-up
    if "Total_Fert_ha" in feats:
        vals = np.linspace(df["Total_Fert_ha"].quantile(0.10),
                           df["Total_Fert_ha"].quantile(0.90), 15)
        rows = []
        for v in vals:
            r = baseline.copy(); r["Total_Fert_ha"] = v
            pred = model.predict(r[feats].values.reshape(1, -1))[0]
            rows.append({"Total_Fert_ha": v, "Predicted_Yield_ha": round(pred, 2)})
        save_csv(pd.DataFrame(rows), TABLES_DIR / "table7_scenario_fert.csv")

    # Scenario C: Climate warming x tercile
    if "Temp_annual" in feats and TERCILE_VAR_MAIN in feats:
        labels = ["T1_Low", "T2_Med", "T3_High"]
        terciles = pd.qcut(df[TERCILE_VAR_MAIN], 3, labels=labels)
        rows = []
        for t_label in labels:
            t_base = build_baseline(df[terciles == t_label], feats)
            for delta in SCENARIO_TEMP_DELTA:
                rb = t_base.copy()
                rb["Temp_annual"] += delta
                if "Temp_grow" in feats: rb["Temp_grow"] += delta
                pred = model.predict(rb[feats].values.reshape(1, -1))[0]
                rows.append({"Irrigation_Tercile": t_label, "Temp_Delta_C": delta,
                             "Predicted_Yield_ha": round(pred, 2)})
        save_csv(pd.DataFrame(rows), TABLES_DIR / "table7_scenario_climate_warming.csv")

    logger.info("Scenarios complete.")


if __name__ == "__main__":
    main()
