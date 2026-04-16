# Reviewer Response Assets

This document maps each reviewer concern to the script and output that addresses it.

## Reviewer 1

| Concern | Script | Output |
|---------|--------|--------|
| Out-of-sample generalisability | 03_validation_5fold_lodo.py | table5_combined_performance.csv |
| District-level benchmark comparison | 09_fixed_effects_benchmark.py | table5b_benchmarks.csv |
| Feature importance not interpretable | 04_shap_analysis.py | table6_shap_importance.csv + figures |

## Reviewer 2

| Concern | Script | Output |
|---------|--------|--------|
| High R2 needs further validation | 08_feature_ablation.py | table5c_ablation.csv |
| Irrigation tercile balance check | 06_tercile_analysis_total_water.py | tableB1_balance_tests.csv |
| Robustness of tercile specification | 07_tercile_analysis_water_intensity.py | tableA1_water_intensity_tercile.csv |
| Climate warming at different management levels | 05_scenario_simulation.py | table7_scenario_climate_warming.csv |
