# -*- coding: utf-8 -*-
"""
Model Loader Singleton Module (Phase 10 / Phase 12)
Centralized loading and lifecycle management for versioned ML model artifacts, scalers,
metadata, and SHAP TreeExplainer instances using structured logging.
"""

import os
import glob
import json
import joblib
from backend.config import PROJECT_ROOT, MODEL_DIR, logger
from model.explainability import create_explainer

_model = None
_scaler = None
_metadata = None
_explainer = None
_loaded = False

def load_ml_artifacts():
    """
    Loads versioned model artifacts, scaler, metadata, and initializes the SHAP explainer ONCE.
    """
    global _model, _scaler, _metadata, _explainer, _loaded
    if _loaded and _model is not None:
        return _model, _scaler, _metadata, _explainer

    try:
        base_dir = os.path.join(PROJECT_ROOT, "backend")
        metadata_files = sorted(glob.glob(os.path.join(MODEL_DIR, "metadata_v2_*.json")))
        
        if metadata_files:
            latest_meta_path = metadata_files[-1]
            with open(latest_meta_path, 'r') as f:
                _metadata = json.load(f)
            version_id = _metadata["model_version"]
            model_path = os.path.join(MODEL_DIR, f"model_{version_id}.joblib")
            scaler_path = os.path.join(MODEL_DIR, f"scaler_{version_id}.joblib")
            _model = joblib.load(model_path)
            _scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            logger.info(f"ML Engine loaded versioned model: {version_id}")
        else:
            _model = joblib.load(os.path.join(base_dir, 'model.joblib'))
            _scaler = joblib.load(os.path.join(base_dir, 'scaler.joblib'))
            _metadata = joblib.load(os.path.join(base_dir, 'metadata.joblib'))
            logger.info("ML Engine loaded legacy model")

        # Initialize SHAP explainer ONCE
        _explainer = create_explainer(_model)
        logger.info("SHAP Explainer initialized successfully")
        _loaded = True

    except Exception as e:
        logger.error(f"ML Engine / SHAP setup error: {e}")
        _model = _scaler = _metadata = _explainer = None
        _loaded = False

    return _model, _scaler, _metadata, _explainer

# Initial load upon module import
load_ml_artifacts()

def get_model():
    return _model

def get_scaler():
    return _scaler

def get_metadata():
    return _metadata

def get_explainer():
    return _explainer
