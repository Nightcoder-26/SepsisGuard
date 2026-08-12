# -*- coding: utf-8 -*-
"""
Predict API Blueprint (Phase 10 / Phase 12)
Handles POST /predict REST endpoint, authentication, rate-limiting, schema validation,
and tracks last successful inference timestamp for health monitoring.
"""

import hmac
from datetime import datetime
from functools import wraps
from flask import Blueprint, request, jsonify
from pydantic import ValidationError

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from backend.config import API_KEY, PREDICT_RATE_LIMIT, logger
from backend.validation.schemas import VitalsInput, sanitize
from backend.ml.inference import run_ml_pipeline, get_model

limiter = Limiter(get_remote_address, default_limits=[], storage_uri="memory://")
predict_bp = Blueprint("predict_bp", __name__)

_last_successful_inference = None

def get_last_successful_inference():
    return _last_successful_inference

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not key or not hmac.compare_digest(str(key), API_KEY):
            logger.warning(f"Unauthorized API request attempt from remote_addr={request.remote_addr}")
            return jsonify({"error": "Unauthorized", "message": "Invalid or missing API key."}), 401
        return f(*args, **kwargs)
    return decorated

@predict_bp.route('/predict', methods=['POST'])
@require_api_key
@limiter.limit(PREDICT_RATE_LIMIT)
def predict():
    global _last_successful_inference
    if get_model() is None:
        logger.error("Predict endpoint invoked while ML model is not loaded")
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data_json = request.get_json(force=True, silent=True)
        if data_json is None:
            return jsonify({"error": "Bad Request", "message": "Malformed or empty JSON payload"}), 400
        
        try:
            vitals_input = VitalsInput(**data_json)
        except ValidationError as err:
            logger.info("Predict input validation failed (422)")
            return jsonify({
                "error": "Unprocessable Entity",
                "message": "Input validation failed",
                "details": err.errors()
            }), 422

        data = vitals_input.model_dump() if hasattr(vitals_input, "model_dump") else vitals_input.dict()
        result = run_ml_pipeline("manual", data, generate_ai=data.get('generate_synthesis', False))
        if result:
            _last_successful_inference = datetime.now().isoformat()
            logger.info(f"Successful model inference executed at {_last_successful_inference}")
            return jsonify(sanitize(result)), 200
        
        logger.error("Inference execution returned empty result")
        return jsonify({"error": "Inference failed"}), 500
    except Exception as e:
        logger.error(f"Internal server error during prediction: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
