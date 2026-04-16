#!/usr/bin/env bash
# run_all.sh — Execute the full pipeline in order.
# Usage: bash scripts/run_all.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Durian Yield ML Pipeline ==="
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
echo "=== Pipeline complete. Results in results/ and manuscript/ ==="
