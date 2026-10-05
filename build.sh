#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "========================================="
echo "   FraudGuard AI - Render Build Script   "
echo "========================================="

echo "==> Upgrading pip..."
python -m pip install --upgrade pip

echo "==> Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "==> Checking for trained model artifacts..."
if [ ! -f "fraud_project/models/best_model.joblib" ]; then
    echo "==> Model artifacts not found. Executing model training pipeline (main.py)..."
    python fraud_project/main.py
else
    echo "==> Model artifacts verified in fraud_project/models/. Skipping re-training."
fi

echo "========================================="
echo "   Build completed successfully!         "
echo "========================================="
