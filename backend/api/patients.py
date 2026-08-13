# -*- coding: utf-8 -*-
"""
Patients & Dashboard API Blueprint (Phase 10 / Phase 12 / Phase 13)
Handles GET /patients, patient CRUD (add/edit), vitals update, assessment history,
clinician notes, workflow status, health monitoring, and static file serving.
"""

import re
from datetime import datetime
from flask import Blueprint, jsonify, request, send_from_directory, abort
from backend.config import FRONTEND_DIR, logger
from backend.api.predict import require_api_key, get_last_successful_inference
from backend.validation.schemas import sanitize, VitalsInput
from backend.services.telemetry import (
    PATIENTS, patient_state, patient_timeline,
    patient_assessments, patient_notes, patient_workflow_status,
    add_patient, update_patient, record_assessment, add_note, set_workflow_status
)
from backend.ml.model_loader import get_model, get_metadata
from backend.ml.inference import run_ml_pipeline
from pydantic import ValidationError

patients_bp = Blueprint("patients_bp", __name__)

# ─── Static / Page Serving ───────────────────────────────────────────────────

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
    Reports application status, model loading state, model version,
    and last successful inference timestamp.
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

# ─── Patient List ─────────────────────────────────────────────────────────────

@patients_bp.route('/patients', methods=['GET'])
@require_api_key
def get_patients():
    """Return full patient state for all active patients."""
    return jsonify(sanitize(patient_state))

# ─── Add Patient ─────────────────────────────────────────────────────────────

@patients_bp.route('/patients', methods=['POST'])
@require_api_key
def create_patient():
    """
    Register a new patient.
    Required body: { pid, name, age, gender }
    Optional:      { bed, room }
    """
    body = request.get_json(force=True, silent=True) or {}

    pid  = (body.get('pid') or '').strip().upper()
    name = (body.get('name') or '').strip()
    age  = body.get('age')
    gender = (body.get('gender') or 'Unknown').strip()
    bed  = (body.get('bed') or '').strip()
    room = (body.get('room') or '').strip()

    # --- Validation ---
    errors = {}
    if not pid:
        errors['pid'] = 'Patient ID is required.'
    elif not re.match(r'^[A-Z0-9\-]{2,16}$', pid):
        errors['pid'] = 'Patient ID must be 2–16 alphanumeric/dash characters.'
    if not name:
        errors['name'] = 'Patient name is required.'
    if age is None:
        errors['age'] = 'Age is required.'
    else:
        try:
            age_int = int(age)
            if not (0 <= age_int <= 120):
                errors['age'] = 'Age must be between 0 and 120.'
        except (ValueError, TypeError):
            errors['age'] = 'Age must be a valid integer.'
    if gender not in ('M', 'F', 'Male', 'Female', 'Unknown', 'Other'):
        gender = 'Unknown'

    if errors:
        return jsonify({'error': 'Validation failed', 'details': errors}), 422

    ok, result = add_patient(pid, {
        'name': name, 'age': int(age), 'gender': gender,
        'bed': bed, 'room': room, 'base_risk': 20
    })

    if not ok:
        return jsonify({'error': 'Conflict', 'message': result}), 409

    logger.info(f"Patient created via API: {pid}")
    return jsonify({'message': f'Patient {pid} created successfully.', 'patient': sanitize(result)}), 201

# ─── Edit Patient Metadata ────────────────────────────────────────────────────

@patients_bp.route('/patients/<pid>', methods=['PATCH'])
@require_api_key
def edit_patient(pid):
    """
    Update editable patient metadata fields.
    Allowed: name, age, gender, bed, room.
    Assessment history and risk results are never modified.
    """
    pid = pid.upper()
    if pid not in PATIENTS:
        return jsonify({'error': 'Not Found', 'message': f"Patient '{pid}' not found."}), 404

    body = request.get_json(force=True, silent=True) or {}
    allowed_fields = {'name', 'age', 'gender', 'bed', 'room'}
    updates = {k: v for k, v in body.items() if k in allowed_fields}

    if not updates:
        return jsonify({'error': 'Bad Request', 'message': 'No valid fields to update.'}), 400

    # Validate age if present
    if 'age' in updates:
        try:
            age_val = int(updates['age'])
            if not (0 <= age_val <= 120):
                return jsonify({'error': 'Validation failed', 'details': {'age': 'Age must be between 0 and 120.'}}), 422
            updates['age'] = age_val
        except (ValueError, TypeError):
            return jsonify({'error': 'Validation failed', 'details': {'age': 'Age must be a valid integer.'}}), 422

    ok, result = update_patient(pid, updates)
    if not ok:
        return jsonify({'error': 'Update failed', 'message': result}), 500

    logger.info(f"Patient metadata edited via API: {pid}")
    return jsonify({'message': f'Patient {pid} updated.', 'patient': sanitize(result)}), 200

# ─── Update Vitals + Run ML Pipeline ─────────────────────────────────────────

@patients_bp.route('/patients/<pid>/vitals', methods=['POST'])
@require_api_key
def update_vitals(pid):
    """
    Update patient vitals and run the full ML pipeline.
    Appends a new independent assessment record to assessment history.
    Does NOT overwrite previous results.
    """
    pid = pid.upper()
    if pid not in PATIENTS:
        return jsonify({'error': 'Not Found', 'message': f"Patient '{pid}' not found."}), 404

    body = request.get_json(force=True, silent=True) or {}

    # Merge submitted vitals with existing patient state for missing fields
    existing = patient_state.get(pid, {})
    merged = {
        'Heart_Rate':       body.get('Heart_Rate',       existing.get('Heart_Rate',       80.0)),
        'Oxygen_Level':     body.get('Oxygen_Level',     existing.get('Oxygen_Level',     98.0)),
        'Temperature':      body.get('Temperature',      existing.get('Temperature',      37.0)),
        'Blood_Pressure':   body.get('Blood_Pressure',   existing.get('Blood_Pressure',   120.0)),
        'Resp_Rate':        body.get('Resp_Rate',        existing.get('Resp_Rate',        16.0)),
        'Age':              body.get('Age',              PATIENTS[pid].get('age',          65)),
        'Infection_Marker': body.get('Infection_Marker', existing.get('Infection_Marker', 0.5)),
        'generate_synthesis': True,
    }

    # Validate using existing schema
    try:
        vitals_input = VitalsInput(**merged)
    except ValidationError as err:
        return jsonify({
            'error': 'Unprocessable Entity',
            'message': 'Vitals validation failed',
            'details': err.errors()
        }), 422

    vitals_data = vitals_input.model_dump() if hasattr(vitals_input, 'model_dump') else vitals_input.dict()

    result = run_ml_pipeline(pid, vitals_data, generate_ai=True)
    if not result:
        return jsonify({'error': 'Inference failed', 'message': 'ML pipeline returned no result.'}), 500

    # Snapshot of vitals submitted for this assessment
    vitals_snapshot = {k: vitals_data[k] for k in [
        'Heart_Rate', 'Temperature', 'Blood_Pressure', 'Resp_Rate',
        'Oxygen_Level', 'Infection_Marker', 'Age'
    ]}

    # Record to assessment history (independent, never overwrites previous)
    assessment_record = record_assessment(pid, result, vitals_snapshot)

    # Update live patient state with latest result + vitals
    patient_state[pid].update({**vitals_snapshot, **result, 'last_updated': datetime.now().isoformat()})

    logger.info(f"Vitals updated and assessment run for {pid}: risk={result['risk_score']:.1f}%")
    return jsonify(sanitize({
        'message': f'Vitals updated and assessment complete for {pid}.',
        'result': result,
        'assessment': assessment_record,
    })), 200

# ─── Assessment History ───────────────────────────────────────────────────────

@patients_bp.route('/patients/<pid>/assessments', methods=['GET'])
@require_api_key
def get_assessments(pid):
    """Return assessment history for a patient (most recent first)."""
    pid = pid.upper()
    if pid not in PATIENTS:
        return jsonify({'error': 'Not Found', 'message': f"Patient '{pid}' not found."}), 404

    history = list(reversed(patient_assessments.get(pid, [])))
    return jsonify(sanitize({'pid': pid, 'count': len(history), 'assessments': history})), 200

# ─── Clinician Notes ──────────────────────────────────────────────────────────

@patients_bp.route('/patients/<pid>/notes', methods=['GET'])
@require_api_key
def get_notes(pid):
    """Return all clinician notes for a patient (most recent first)."""
    pid = pid.upper()
    if pid not in PATIENTS:
        return jsonify({'error': 'Not Found', 'message': f"Patient '{pid}' not found."}), 404
    notes = list(reversed(patient_notes.get(pid, [])))
    return jsonify(sanitize({'pid': pid, 'count': len(notes), 'notes': notes})), 200

@patients_bp.route('/patients/<pid>/notes', methods=['POST'])
@require_api_key
def create_note(pid):
    """Append a clinician note for a patient. Notes are never deleted or overwritten."""
    pid = pid.upper()
    if pid not in PATIENTS:
        return jsonify({'error': 'Not Found', 'message': f"Patient '{pid}' not found."}), 404

    body = request.get_json(force=True, silent=True) or {}
    text = (body.get('text') or '').strip()
    author = (body.get('author') or 'Clinician').strip()

    if not text:
        return jsonify({'error': 'Validation failed', 'details': {'text': 'Note text is required.'}}), 422
    if len(text) > 2000:
        return jsonify({'error': 'Validation failed', 'details': {'text': 'Note must not exceed 2000 characters.'}}), 422

    note = add_note(pid, text, author)
    logger.info(f"Clinician note added for {pid}")
    return jsonify(sanitize({'message': 'Note saved.', 'note': note})), 201

# ─── Workflow Status ──────────────────────────────────────────────────────────

@patients_bp.route('/patients/<pid>/status', methods=['PATCH'])
@require_api_key
def update_status(pid):
    """
    Update the workflow review status for a patient.
    Valid values: 'Needs Review', 'Under Observation', 'Reviewed'.
    This is a clinical workflow concept — distinct from ML risk level.
    """
    pid = pid.upper()
    if pid not in PATIENTS:
        return jsonify({'error': 'Not Found', 'message': f"Patient '{pid}' not found."}), 404

    body = request.get_json(force=True, silent=True) or {}
    status = (body.get('status') or '').strip()

    ok, result = set_workflow_status(pid, status)
    if not ok:
        return jsonify({'error': 'Validation failed', 'details': {'status': result}}), 422

    logger.info(f"Workflow status updated for {pid}: {status}")
    return jsonify({'message': f'Status updated.', 'pid': pid, 'workflow_status': result}), 200

# ─── Static file catch-all ────────────────────────────────────────────────────

@patients_bp.route('/<path:filename>')
def serve_static(filename):
    if filename.startswith('socket.io'):
        abort(404)
    return send_from_directory(FRONTEND_DIR, filename)
