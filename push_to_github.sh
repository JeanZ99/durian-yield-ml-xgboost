#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "============================================================================"
echo "Updating GitHub Repository: https://github.com/JeanZ99/durian-yield-ml-xgboost"
echo "============================================================================"
if [ ! -d ".git" ]; then
    git init
    git branch -M main
    git remote add origin https://github.com/JeanZ99/durian-yield-ml-xgboost.git
else
    git remote set-url origin https://github.com/JeanZ99/durian-yield-ml-xgboost.git
fi
git add .
git commit -m "Update ML pipeline, dataset, and reproduction outputs for manuscript v14"
git push -u origin main --force
echo "Done! Please check https://github.com/JeanZ99/durian-yield-ml-xgboost"
