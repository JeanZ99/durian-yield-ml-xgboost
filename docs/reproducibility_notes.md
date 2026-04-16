# Reproducibility Notes

## Environment
All analyses were conducted using Python 3.11 on Ubuntu 22.04.
The exact software environment is captured in `environment.yml` and `requirements.txt`.

## Execution order
Run scripts in numerical order (01 → 10). Each script depends on outputs of prior scripts.

```bash
bash scripts/run_all.sh
```

Or individually:
```bash
python src/01_data_prep.py
python src/02_train_xgboost.py --n-trials 100
python src/03_validation_5fold_lodo.py
python src/04_shap_analysis.py
python src/05_scenario_simulation.py
python src/06_tercile_analysis_total_water.py
python src/07_tercile_analysis_water_intensity.py
python src/08_feature_ablation.py
python src/09_fixed_effects_benchmark.py
python src/10_make_tables_figures.py
```

## Random seeds
All random seeds are fixed at `RANDOM_SEED = 42` in `src/config.py`.
Optuna hyperparameter search uses `TPESampler(seed=42)`.

## Data access
The file `Cleaned_Durian_Data_ML.xlsx` contains farm-level data from 393
GAP-certified durian farms surveyed in Chanthaburi Province, Thailand (2021–2025).
Contact the corresponding author for data access subject to confidentiality approval.
