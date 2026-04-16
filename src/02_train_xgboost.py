"""
02_train_xgboost.py
Optuna hyperparameter search + final model training.
Run: python src/02_train_xgboost.py [--n-trials 100]
"""
import sys, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from config import (DATA_PROC, MODELS_DIR, TABLES_DIR, LOGS_DIR,
                    FEATURES_ALL, TARGET, CV_FOLDS, RANDOM_SEED)
from utils import get_logger, ensure_dirs, regression_metrics, save_csv

logger = get_logger("02_train_xgboost", LOGS_DIR)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    df = pd.read_parquet(DATA_PROC / "durian_ml_processed.parquet")
    feats = [f for f in FEATURES_ALL if f in df.columns]
    return df[feats].values, df[TARGET].values, feats


def objective(trial, X, y):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "random_state": RANDOM_SEED, "n_jobs": -1, "tree_method": "hist",
    }
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for tr, val in kf.split(X):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
        scores.append(r2_score(y[val], m.predict(X[val])))
    return np.mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
    args = parser.parse_args()
    ensure_dirs(MODELS_DIR, TABLES_DIR, LOGS_DIR)

    X, y, feats = load_data()
    logger.info(f"Features: {len(feats)}, N={len(y)}")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda t: objective(t, X, y),
                   n_trials=args.n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"random_state": RANDOM_SEED, "n_jobs": -1, "tree_method": "hist"})
    logger.info(f"Best CV R2: {study.best_value:.4f}")

    save_csv(study.trials_dataframe(), TABLES_DIR / "optuna_trials.csv")

    final = xgb.XGBRegressor(**best_params)
    final.fit(X, y)
    logger.info(f"Train metrics: {regression_metrics(y, final.predict(X))}")

    final.save_model(str(MODELS_DIR / "xgb_final.json"))
    (MODELS_DIR / "xgb_params.json").write_text(json.dumps(best_params, indent=2))
    (MODELS_DIR / "feature_list.txt").write_text("\n".join(feats))
    logger.info("Model, params, feature list saved.")


if __name__ == "__main__":
    main()
