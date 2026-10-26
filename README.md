# Predicting Yield Variation Among GAP-Certified Durian Farms in Chanthaburi, Thailand: The Roles of Orchard Structure and Management

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19632851.svg)](https://doi.org/10.5281/zenodo.19632851)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.44-purple.svg)

This repository contains the complete analytical pipeline, feature-screening audit scripts, spatial validation routines, feature-block ablations, and TreeSHAP attribution scripts for the manuscript:

> **Phaiboontrirat, J., Tsusaka, T.W.\*, Himanshu, S.K., Pramanik, M.K.** *Predicting yield variation among GAP-certified durian farms in Chanthaburi, Thailand: The roles of orchard structure and management.*

---

## 1. Overview of the Analytical Pipeline

The study analyzes primary survey and orchard topsoil data from **$n = 393$ GAP-certified durian farms** across all ten districts of Chanthaburi Province, Thailand (2024–2025 production season, collected in 2025).

### Key Methodological Specifications
- **Pre-Modeling Screening & Leakage Audit (`reanalysis_durian.py`):**
  - Excludes legacy yield-derived intensity ratios (`Water_intensity = Total_water_ha / Yield_ha` and `Fert_intensity = Total_Fert_ha / Yield_ha`) where observed yield enters the denominator.
  - Reconstructs yield-independent per-tree input measures (`water_per_productive_tree = Total_water_ha / Productive_tree_ha`, `fertilizer_per_productive_tree = Total_Fert_ha / Productive_tree_ha`, `organic_fertilizer_per_productive_tree = Total_OrgFert_ha / Productive_tree_ha`).
  - Excludes post-outcome accounting/revenue variables (`Total_yield`, `Total_Income`, `Total_Income_ha`, `Revenue_ha`), market prices (`Price`, unit fertilizer prices), monetary cost fields (`Cp_ha`, `Cf_ha`, etc.), and zero-variance provincial weather/irrigation-type fields (`Temp_grow`, `Temp_annual`, `Rain_grow`, `Rain_annual`, `Primary_Irrigation`, `Modern_Irrigation`).
- **Full Revised Specification (33 Predictors, `run_leakage_free_model.py`):**
  - Comprises 14 management-input variables (7 physical/per-tree input quantities + 7 management/institutional indicators), 7 orchard-structure variables, 5 farmer-characteristic variables, and 7 soil-context variables.
  - Evaluated via shuffled 5-fold cross-validation ($n = 393$, $R^2 = 0.968$, $\text{RMSE} = 253.5\text{ kg ha}^{-1}$) and leave-one-district-out (LODO) spatial cross-validation across the 8 districts with $n \ge 5$ ($n = 389$, $R^2 = 0.967$, $\text{RMSE} = 257.8\text{ kg ha}^{-1}$).
- **Feature-Block Ablation & Permutation Placebo (`run_ablation_and_robustness.py`):**
  - Evaluates 5 pre-specified feature-block removals and a permuted-outcome placebo (`random_state = 42`) under unchanged hyperparameters.
- **Reduced Specification (26 Predictors, `run_redundancy_diagnostics.py` & `run_deredundant_robustness.py`):**
  - Removes 7 highly collinear predictors ($|r| \ge 0.85$): `Tree_Density_ha`, `N`, `water_per_productive_tree`, `Total_Fert_ha`, `Total_OrgFert_ha`, `fertilizer_per_productive_tree`, and `organic_fertilizer_per_productive_tree`.
  - Retains comparable accuracy (5-fold $R^2 = 0.969$; LODO $R^2 = 0.968$) and evaluates sensitivity to removing farming experience (`Experience_years`, 25 predictors: 5-fold $R^2 = 0.931$; LODO $R^2 = 0.928$).
- **Model Interpretation via TreeSHAP (`run_deredundant_shap_ale.py`):**
  - Fits the 26-predictor reduced XGBoost model to the full median-imputed sample ($n = 393$) and computes exact `shap.TreeExplainer` attributions at the individual predictor level (Table 7, Fig. 4) and domain-block level (Table 8).

---

## 2. Repository Structure

```text
durian-yield-ml-xgboost/
├── data/
│   ├── Cleaned_Durian_Data_ML.xlsx         # Primary analytical dataset (n = 393 farms)
│   └── Data-Dictionary-ML.xlsx             # Variable definitions, units, and coding
├── results/
│   ├── tables/                             # Consolidated CSV tables matching Tables 2, 4, 5, 6, 7, 8
│   ├── output/                             # Pre-modeling audit diagnostics (Table 3)
│   ├── output_revised_model/               # 33-predictor Full Revised Model outputs (Tables 2 & 4)
│   ├── output_ablation/                    # Feature-block ablation & placebo outputs (Table 5)
│   ├── output_redundancy/                  # Correlation matrices & high-redundancy pairs (|r| >= 0.85)
│   ├── output_reduced_model/               # 26-predictor Reduced Model outputs (Table 6)
│   └── output_reduced_shap/                # Reduced-model TreeSHAP outputs (Tables 7 & 8, Fig. 4)
├── reanalysis_durian.py                    # Step 1: Pre-modeling leakage & variance audit
├── run_leakage_free_model.py               # Step 2: Full revised model (33 predictors) 5-fold CV & LODO
├── run_ablation_and_robustness.py          # Step 3: Feature-block ablations & outcome permutation
├── run_redundancy_diagnostics.py           # Step 4: Pairwise Pearson/Spearman correlation screening
├── run_deredundant_robustness.py           # Step 5: Reduced model (26 predictors) & minus-experience check
├── run_deredundant_shap_ale.py             # Step 6: TreeSHAP feature & block attributions (26 predictors)
├── run_all.py                              # Master script to run all 6 steps and consolidate tables
├── requirements.txt                        # Python package dependencies
├── CITATION.cff                            # Citation metadata
├── LICENSE                                 # MIT (code) / CC-BY 4.0 (data)
└── README.md                               # Documentation
```

---

## 3. Quick Start & Reproduction Instructions

### 3.1 Install Dependencies
Ensure Python 3.11+ is installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 3.2 Run the Entire Pipeline (All Tables & Figures)
Run the master script from the repository root:

```bash
python run_all.py
```

Or run any individual script directly:

```bash
python reanalysis_durian.py
python run_leakage_free_model.py
python run_ablation_and_robustness.py
python run_redundancy_diagnostics.py
python run_deredundant_robustness.py
python run_deredundant_shap_ale.py
```

---

## 4. Mapping Between Manuscript Tables/Figures and Repository Outputs

| Manuscript Item | Script | Generated Output File | Key Verified Result |
| :--- | :--- | :--- | :--- |
| **Table 2** (Descriptive statistics, $n = 393$) | `run_leakage_free_model.py` | `results/tables/Table2_descriptive_statistics.csv` | Mean yield = $11,371.0\text{ kg ha}^{-1}$ ($\text{SD} = 1,428.5$) |
| **Table 3** (Pre-modeling audit & exclusions) | `reanalysis_durian.py` | `results/output/water_intensity_diagnostic.csv`<br>`results/output/temperature_by_district.csv` | Confirms `Water_intensity = Total_water_ha / Yield_ha` ($r = 1.0$) and constant `Temp_grow = 29.6°C` |
| **Table 4 & Fig. 2** (LODO performance by district, $n = 389$) | `run_leakage_free_model.py` | `results/tables/Table4_lodo_by_district.csv`<br>`results/output_revised_model/fivefold_overall_metrics.csv`<br>`results/output_revised_model/lodo_overall_metrics.csv` | 5-fold $R^2 = 0.968$ ($\text{RMSE} = 253.5$); LODO $R^2 = 0.967$ ($\text{RMSE} = 257.8$); district $R^2$ range $0.947$–$0.976$ |
| **Table 5 & Fig. 3** (Feature-block ablation & placebo) | `run_ablation_and_robustness.py` | `results/tables/Table5_feature_ablation_and_placebo.csv` | Minus water LODO $R^2 = 0.965$; Minus orchard $R^2 = 0.890$; Minus mgmt+orchard $R^2 = 0.399$; Placebo $R^2 = -0.170$ |
| **Table 6** (Reduced model & experience check) | `run_deredundant_robustness.py` | `results/tables/Table6_reduced_model_robustness.csv` | Reduced (26 vars): 5-fold $R^2 = 0.969$, LODO $R^2 = 0.968$; Minus experience (25 vars): 5-fold $R^2 = 0.931$, LODO $R^2 = 0.928$ |
| **Table 7 & Fig. 4** (Leading predictors by TreeSHAP) | `run_deredundant_shap_ale.py` | `results/tables/Table7_reduced_model_shap_features.csv` | Tree age: $976.5\text{ kg ha}^{-1}$ ($64.96\%, r = 0.994$); Total irrigation: $239.1\text{ kg ha}^{-1}$ ($15.90\%, r = 0.982$); Experience: $163.5\text{ kg ha}^{-1}$ ($10.87\%, r = -0.864$) |
| **Table 8** (SHAP attribution by predictor block) | `run_deredundant_shap_ale.py` | `results/tables/Table8_reduced_model_shap_blocks.csv` | Orchard structure $65.65\%$, Management inputs $18.24\%$, Farmer characteristics $12.34\%$, Soil context $3.77\%$ |

---

## 5. Fixed Hyperparameters & Reproducibility Settings

All models use identical fixed hyperparameters (`RANDOM_STATE = 42`) to ensure that performance comparisons reflect predictor sets rather than hyperparameter retuning:

```python
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.01,
    "reg_lambda": 5,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0
}
```
