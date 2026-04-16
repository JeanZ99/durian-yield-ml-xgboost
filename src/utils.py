"""
utils.py — Shared helper functions.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_logger(name: str, log_dir: Path = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / f"{name}.log")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2":   round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 2),
    }


def load_clean_data(filepath: Path, target: str, features: list,
                    group_col: str = None) -> tuple:
    df = pd.read_excel(filepath)
    keep = features + [target]
    if group_col and group_col not in features:
        keep.append(group_col)
    df = df[[c for c in keep if c in df.columns]].dropna(subset=[target]).reset_index(drop=True)
    if "Primary_Irrigation" in df.columns:
        df["Primary_Irrigation"] = df["Primary_Irrigation"].astype("category").cat.codes
    feats_avail = [f for f in features if f in df.columns]
    X = df[feats_avail]
    y = df[target].values
    groups = df[group_col].values if group_col else None
    return X, y, groups


def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def print_metrics(label: str, metrics: dict):
    print(f"  [{label}]  R2={metrics['R2']:.4f}  "
          f"RMSE={metrics['RMSE']:,.1f}  MAE={metrics['MAE']:,.1f}")


def save_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved -> {path}")
