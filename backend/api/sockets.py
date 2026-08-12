# -*- coding: utf-8 -*-
"""
Socket.IO Event Handlers & Room Scoping (Phase 10 / Phase 12)
Manages real-time Socket.IO authentication, room scoping ('icu_unit_a'), telemetry snapshots,
AI synthesis requests, and AI Copilot queries using structured logging.
"""

import hmac
from flask import request
from flask_socketio import emit, join_room
from backend.config import API_KEY, logger
from backend.validation.schemas import sanitize
from backend.ml.inference import run_ml_pipeline
from backend.services.telemetry import PATIENTS, patient_state, patient_timeline, start_telemetry_thread
from backend.services.copilot import copilot_answer

def register_socket_events(socketio):

    @socketio.on('connect')
    def on_connect(auth=None):
        token = None
        if isinstance(auth, dict):
            token = auth.get('token')
        elif request.args.get('token'):
            token = request.args.get('token')

        if not token or not hmac.compare_digest(str(token), API_KEY):
            logger.warning(f"Socket connection rejected for {request.sid}: Invalid or missing auth token.")
            return False

        join_room('icu_unit_a')
        logger.info(f"Dashboard connected and authenticated: {request.sid} (Room: icu_unit_a)")
        
        emit('snapshot', sanitize({pid: patient_state[pid] for pid in PATIENTS}), room=request.sid)
        emit('timeline_snapshot', sanitize({pid: patient_timeline[pid] for pid in PATIENTS}), room=request.sid)
        
        start_telemetry_thread(socketio)

    @socketio.on('disconnect')
    def on_disconnect():
        logger.info(f"Dashboard disconnected: {request.sid}")

    @socketio.on('request_ai_synthesis')
    def on_request_ai(data):
        pid = data.get('pid')
        if pid and pid in patient_state:
            vitals = {k: patient_state[pid][k] for k in [
                'Heart_Rate', 'Temperature', 'Blood_Pressure',
                'Resp_Rate', 'Oxygen_Level', 'Age', 'Infection_Marker'
            ]}
            result = run_ml_pipeline(pid, vitals, generate_ai=True)
            if result:
                patient_state[pid].update(result)
                emit('ai_synthesis_result', sanitize({"pid": pid, **result}), room=request.sid)

    @socketio.on('copilot_query')
    def on_copilot(data):
        question = data.get('question', '')
        pid      = data.get('pid')

        if pid and pid in patient_state:
            ps = patient_state[pid]
            context = (
                f"Patient: {ps['name']}, Bed: {ps['bed']}, Age: {ps['age']}. "
                f"Vitals: HR={ps.get('Heart_Rate',0):.0f}, Temp={ps.get('Temperature',0):.1f}C, "
                f"SysBP={ps.get('Blood_Pressure',0):.0f}, RR={ps.get('Resp_Rate',0):.0f}, "
                f"SpO2={ps.get('Oxygen_Level',0):.1f}%, InfMkr={ps.get('Infection_Marker',0):.2f}. "
                f"Sepsis risk: {ps.get('risk_score',0):.0f}% ({ps.get('risk_level','?')}). "
                f"Triggers: {', '.join(ps.get('explanation', []))}. "
                f"AI note: {ps.get('ai_synthesis', '')}"
            )
        else:
            lines = []
            for p, ps in patient_state.items():
                lines.append(f"{ps['name']} ({ps['bed']}): risk {ps.get('risk_score',0):.0f}%, {ps.get('alert_level','?')}")
            context = "ICU Overview: " + " | ".join(lines)

        answer = copilot_answer(question, context)
        emit('copilot_response', {"answer": answer, "pid": pid}, room=request.sid)
