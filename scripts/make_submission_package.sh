#!/usr/bin/env bash
# make_submission_package.sh — Zip submission-ready assets.
# Usage: bash scripts/make_submission_package.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ZIP="durian_ml_submission_$(date +%Y%m%d).zip"
zip -r "$ZIP" \
    manuscript/tables_for_submission/ \
    manuscript/figures_for_submission/ \
    README.md

echo "Package ready: $ZIP"
