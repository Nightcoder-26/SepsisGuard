# -*- coding: utf-8 -*-
"""
Patients & Dashboard API Blueprint (Phase 10 / Phase 12)
Handles GET /patients, enhanced GET /health monitoring, static file serving, and patient queries.
"""

from flask import Blueprint, jsonify, send_from_directory, abort
from backend.config import FRONTEND_DIR, logger
from backend.api.predict import require_api_key, get_last_successful_inference
from backend.validation.schemas import sanitize
from backend.services.telemetry import PATIENTS, patient_state
from backend.ml.model_loader import get_model, get_metadata

patients_bp = Blueprint("patients_bp", __name__)

@patients_bp.route('/')
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, 'dashboard.html')

@patients_bp.route('/patient')
def serve_patient():
    return send_from_directory(FRONTEND_DIR, 'patient.html')

@patients_bp.route('/health', methods=['GET'])
def health():
    """
    Enhanced Health Check Endpoint:
    Reports application status, model loading state, model version, and last successful inference timestamp.
    """
    model = get_model()
    metadata = get_metadata()
    model_loaded = model is not None
    model_version = metadata.get("model_version", "v2_2026-08-12") if metadata else "v2_2026-08-12"
    status_str = "ok" if model_loaded else "degraded"

    return jsonify({
        "status": status_str,
        "model_loaded": model_loaded,
        "model_version": model_version,
        "last_successful_inference": get_last_successful_inference(),
        "active_patients": len(PATIENTS),
        "service": "SepsisGuard AI v3.0 Telemetry Engine"
    }), (200 if model_loaded else 503)

@patients_bp.route('/patients', methods=['GET'])
@require_api_key
def get_patients():
    return jsonify(sanitize(patient_state))

@patients_bp.route('/<path:filename>')
def serve_static(filename):
    if filename.startswith('socket.io'):
        abort(404)
    return send_from_directory(FRONTEND_DIR, filename)
