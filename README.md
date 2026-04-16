# Does Farm Management Dominate Climate in Determining Yields of GAP-Certified Durian?
## An Interpretable Machine Learning Approach

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the analytical code, reproducibility files, and research
assets for the manuscript:

> **Phaiboontrirat, J., Tsusaka, T. W., Himanshu, S., & Pramanik, M.**
> *Does Farm Management Dominate Climate in Determining Yields of GAP-Certified Durian?*
> *An Interpretable Machine Learning Approach.*
> *(Under review)*

---

## Overview

This study applies an interpretable XGBoost framework to farm-level durian yield
prediction using survey data from **393 GAP-certified farms** in Chanthaburi Province,
Thailand (2021–2025). Key contributions:

- Leakage-free 5-fold and leave-one-district-out (LODO) validation
- SHAP-based feature attribution across seven management, climate, and soil blocks
- Counterfactual scenario simulation for irrigation step-ups and climate warming
- Irrigation-tercile climate sensitivity analysis with bootstrap confidence intervals
- Feature ablation study to quantify each block's contribution to predictive accuracy
- District fixed-effects OLS benchmarks for comparison

---

## Repository Structure

```
durian-yield-ml-xgboost/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── environment.yml
├── requirements.txt
├── data/
│   ├── README.md              <- Data access policy
│   ├── raw/                   <- Original data files (restricted + public)
│   ├── interim/               <- Intermediate processing outputs
│   └── processed/             <- Final Parquet files for modelling
├── src/
│   ├── config.py              <- Central configuration (paths, features, hyperparameters)
│   ├── utils.py               <- Shared helper functions
│   ├── 01_data_prep.py        <- Load, clean, engineer features, save Parquet
│   ├── 02_train_xgboost.py    <- Optuna hyperparameter search + final model training
│   ├── 03_validation_5fold_lodo.py   <- 5-fold CV + LODO validation (Table 5)
│   ├── 04_shap_analysis.py    <- SHAP importance + beeswarm + PDP + interaction plots
│   ├── 05_scenario_simulation.py     <- Counterfactual water/fertilizer/climate scenarios
│   ├── 06_tercile_analysis_total_water.py   <- Table 8 + Appendix B balance tests
│   ├── 07_tercile_analysis_water_intensity.py  <- Appendix A robustness check
│   ├── 08_feature_ablation.py        <- Table 5c feature ablation study
│   ├── 09_fixed_effects_benchmark.py <- Table 5b OLS district FE benchmarks
│   └── 10_make_tables_figures.py     <- Assemble submission-ready tables and figures
├── notebooks/
│   ├── 01_eda.ipynb           <- Exploratory data analysis
│   └── 02_model_diagnostics.ipynb   <- Residuals, predicted vs. actual
├── results/
│   ├── tables/                <- All CSV outputs from src/ scripts
│   ├── figures/               <- All PNG figures
│   ├── models/                <- Saved XGBoost model and hyperparameters
│   └── logs/                  <- Execution logs
├── manuscript/
│   ├── tables_for_submission/ <- Final formatted tables (Table 5–8, A1, B1)
│   └── figures_for_submission/ <- 300 DPI TIFF figures for journal submission
├── docs/
│   ├── reproducibility_notes.md
│   ├── variable_mapping.md    <- Manuscript labels to code variable names
│   └── reviewer_response_assets.md
└── scripts/
    ├── run_all.sh             <- Run complete pipeline
    └── make_submission_package.sh  <- Zip submission assets
```

---

## Output Map

| Manuscript Item | Script | Output File |
|-----------------|--------|-------------|
| Table 5 — Model performance | `03_validation_5fold_lodo.py` | `table5_combined_performance.csv` |
| Table 5b — OLS benchmarks | `09_fixed_effects_benchmark.py` | `table5b_benchmarks.csv` |
| Table 5c — Feature ablation | `08_feature_ablation.py` | `table5c_ablation.csv` |
| Table 6 — SHAP importance | `04_shap_analysis.py` | `table6_shap_importance.csv` |
| Table 7 — Scenario simulation | `05_scenario_simulation.py` | `table7_scenario_climate_warming.csv` |
| Table 8 — Tercile × climate | `06_tercile_analysis_total_water.py` | `table8_tercile_total_water.csv` |
| Appendix A — Robustness | `07_tercile_analysis_water_intensity.py` | `tableA1_water_intensity_tercile.csv` |
| Appendix B — Balance tests | `06_tercile_analysis_total_water.py` | `tableB1_balance_tests.csv` |
| Figure 2 — SHAP beeswarm | `04_shap_analysis.py` | `shap_beeswarm.png` |
| Figure 3 — SHAP bar | `04_shap_analysis.py` | `shap_bar_top20.png` |
| Figure 4 — PDP irrigation | `04_shap_analysis.py` | `shap_pdp_total_water.png` |
| Figure 5 — Interaction | `04_shap_analysis.py` | `shap_interaction_water_temp.png` |

---

## Data Availability

Raw farm survey data are **not publicly available** due to confidentiality constraints.
The data dictionary (`Data-Dictionary-ML.xlsx`) is included in `data/raw/`.

Researchers interested in data access may contact the corresponding author.

---

## Requirements

- Python 3.11
- XGBoost >= 2.0
- scikit-learn >= 1.4
- SHAP >= 0.44
- pandas, numpy, statsmodels, matplotlib, seaborn, openpyxl, optuna, pyarrow

### Install with conda (recommended)

```bash
conda env create -f environment.yml
conda activate durian-ml
```

### Install with pip

```bash
pip install -r requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/durian-yield-ml-xgboost.git
cd durian-yield-ml-xgboost

# Copy your data files into data/raw/
cp /path/to/Cleaned_Durian_Data_ML.xlsx data/raw/
cp /path/to/Data-Dictionary-ML.xlsx data/raw/

# Run the full pipeline
bash scripts/run_all.sh
```

Results appear in `results/tables/`, `results/figures/`, and `results/models/`.
Submission-ready assets are assembled in `manuscript/tables_for_submission/` and
`manuscript/figures_for_submission/`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{phaiboontrirat2026durian,
  author  = {Phaiboontrirat, Jintawat and Tsusaka, Takuji W. and
             Himanshu, Shilpi and Pramanik, Md},
  title   = {Does Farm Management Dominate Climate in Determining Yields
             of {GAP}-Certified Durian? {A}n Interpretable Machine Learning Approach},
  journal = {(under review)},
  year    = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
