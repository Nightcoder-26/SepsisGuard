# -*- coding: utf-8 -*-
"""
SepsisGuard v3.0 - Clean-Clone Smoke Test Script
Verifies environment setup, dependency imports, model loading, SHAP explainer initialization,
and inference pipeline health on a fresh clone.
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_smoke_test():
    print("=" * 70)
    print("SEPSISGUARD v3.0 - REPRODUCIBILITY & SMOKE TEST SUITE")
    print("=" * 70)

    # 1. Dependency Imports Check
    print("[*] Checking core dependency imports...")
    try:
        import flask
        import flask_cors
        import flask_socketio
        import flask_limiter
        import pydantic
        import dotenv
        import requests
        import sklearn
        import pandas
        import numpy
        import joblib
        import matplotlib
        import seaborn
        import shap
        import xgboost
        print("[OK] Core dependencies imported successfully.")
    except ImportError as err:
        print(f"[FAIL] Missing dependency: {err}")
        sys.exit(1)

    # 2. Environment Setup Check
    print("[*] Checking environment configuration...")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, "backend", ".env"))
    
    secret = os.environ.get("FLASK_SECRET_KEY")
    api_key = os.environ.get("API_KEY")
    if not secret:
        print("[WARNING] FLASK_SECRET_KEY is missing from environment. Using test default for smoke check.")
        os.environ["FLASK_SECRET_KEY"] = "smoke_test_secret_key_12345"
    if not api_key:
        print("[WARNING] API_KEY is missing from environment. Using test default for smoke check.")
        os.environ["API_KEY"] = "sepsisguard_api_key_3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b"
    print("[OK] Environment variables verified.")

    # 3. Model & Metadata Artifact Reloading
    print("[*] Verifying model artifacts...")
    model_dir = os.path.join(PROJECT_ROOT, "model")
    metadata_files = sorted([f for f in os.listdir(model_dir) if f.startswith("metadata_v2_") and f.endswith(".json")])
    
    if not metadata_files:
        print("[FAIL] No versioned metadata artifact found in model/")
        sys.exit(1)

    latest_meta = metadata_files[-1]
    import json
    with open(os.path.join(model_dir, latest_meta), 'r') as f:
        meta_data = json.load(f)

    version_id = meta_data["model_version"]
    model_path = os.path.join(model_dir, f"model_{version_id}.joblib")
    scaler_path = os.path.join(model_dir, f"scaler_{version_id}.joblib")

    if not os.path.exists(model_path):
        print(f"[FAIL] Model artifact missing: {model_path}")
        sys.exit(1)

    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    print(f"[OK] Loaded model artifact: model_{version_id}.joblib")
    print(f"[OK] Loaded model metadata: {version_id} (Model Type: {meta_data['model_type']})")

    # 4. SHAP Explainer Check
    print("[*] Testing SHAP TreeExplainer initialization...")
    from model.explainability import create_explainer, explain_prediction
    explainer = create_explainer(loaded_model)
    if explainer is None:
        print("[FAIL] SHAP TreeExplainer initialization failed!")
        sys.exit(1)
    print("[OK] SHAP TreeExplainer initialized successfully.")

    # 5. Pipeline Test Prediction
    print("[*] Executing test prediction pipeline...")
    from backend.app import run_ml_pipeline
    sample_vitals = {
        "Heart_Rate": 115.0,
        "Oxygen_Level": 92.0,
        "Temperature": 38.8,
        "Blood_Pressure": 88.0,
        "Resp_Rate": 24.0,
        "Age": 68.0,
        "Infection_Marker": 0.85
    }
    result = run_ml_pipeline("smoke_test_patient", sample_vitals)
    if not result or "risk_score" not in result or "shap_explanation" not in result:
        print("[FAIL] ML pipeline execution failed or missing expected output keys.")
        sys.exit(1)
    
    print(f"[OK] Inference prediction successful. Risk Score: {result['risk_score']}%, Level: {result['risk_level']}")
    print(f"[OK] SHAP Explanation available: {result['shap_explanation'].get('available', False)}")

    # 6. Dataset Availability Check
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    test_csv = os.path.join(processed_dir, "test.csv")
    if os.path.exists(test_csv):
        print(f"[OK] Processed dataset split available: {test_csv}")
    else:
        print("[NOTE] Processed dataset split not found locally. Follow data/DATASET_CARD.md to obtain data.")

    print("=" * 70)
    print("[ALL PASSED] SepsisGuard v3.0 Clean-Clone Smoke Test PASSED cleanly.")
    print("=" * 70)

if __name__ == '__main__':
    run_smoke_test()
