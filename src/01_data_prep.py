"""
01_data_prep.py
Load raw Excel, engineer derived features, and save processed Parquet.
Run: python src/01_data_prep.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import (RAW_DATA_FILE, DATA_PROC, TARGET, FEATURES_ALL, GROUP_CV, LOGS_DIR)
from utils import get_logger, ensure_dirs

logger = get_logger("01_data_prep", LOGS_DIR)


def main():
    ensure_dirs(DATA_PROC, LOGS_DIR)
    logger.info(f"Loading {RAW_DATA_FILE}")
    df = pd.read_excel(RAW_DATA_FILE)
    logger.info(f"Raw shape: {df.shape}")

    if "Water_intensity" not in df.columns:
        df["Water_intensity"] = df["Total_water_ha"] / df["Land"].replace(0, np.nan)

    if "Fert_intensity" not in df.columns:
        df["Fert_intensity"] = df["Total_Fert_ha"] / df["Land"].replace(0, np.nan)

    if "Soil_fertility_index" not in df.columns:
        soil_cols = [c for c in ["PH", "OM", "N"] if c in df.columns]
        if soil_cols:
            z = df[soil_cols].apply(lambda s: (s - s.mean()) / s.std())
            df["Soil_fertility_index"] = z.mean(axis=1)

    if "Primary_Irrigation" in df.columns:
        df["Primary_Irrigation"] = df["Primary_Irrigation"].astype("category").cat.codes

    before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} rows with missing {TARGET}")
    logger.info(f"N={len(df)} | Yield range: {df[TARGET].min():.1f}–{df[TARGET].max():.1f} kg/ha")

    keep_cols = list(dict.fromkeys(
        ["ID", GROUP_CV, TARGET] + [f for f in FEATURES_ALL if f in df.columns]
    ))
    df_out = df[[c for c in keep_cols if c in df.columns]]

    out = DATA_PROC / "durian_ml_processed.parquet"
    df_out.to_parquet(out, index=False)
    logger.info(f"Saved -> {out}  shape={df_out.shape}")

    df_out.describe().T.to_csv(DATA_PROC / "descriptive_stats.csv")
    logger.info("Descriptive stats saved.")


if __name__ == "__main__":
    main()
