# -*- coding: utf-8 -*-
"""
SepsisGuard AI - Central Hospital Telemetry Server v3.0
Real-time ICU Intelligence Ecosystem
"""

import sys
import warnings
warnings.filterwarnings('ignore')  # suppress sklearn warnings

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import joblib
import numpy as np
import os
import requests
import threading
import time
import random
import math
from datetime import datetime
from dotenv import load_dotenv

def sanitize(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):   return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):   return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sepsisguard-icu-v3'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False, ping_timeout=60, ping_interval=25)

# ─────────────────────────────────────────────
# Load ML Model
# ─────────────────────────────────────────────
try:
    BASE     = os.path.dirname(__file__)
    model    = joblib.load(os.path.join(BASE, 'model.joblib'))
    scaler   = joblib.load(os.path.join(BASE, 'scaler.joblib'))
    metadata = joblib.load(os.path.join(BASE, 'metadata.joblib'))
    print("[OK] ML Engine loaded")
except Exception as e:
    print(f"[FAIL] ML Engine: {e}")
    model = scaler = metadata = None

# ─────────────────────────────────────────────
# Patient Registry
# ─────────────────────────────────────────────
PATIENTS = {
    "P001": {"name": "James Hartwell",  "age": 67, "bed": "ICU-01", "room": "A1", "base_risk": 15,  "gender": "M"},
    "P002": {"name": "Sarah Chen",      "age": 45, "bed": "ICU-02", "room": "A2", "base_risk": 72,  "gender": "F"},
    "P003": {"name": "Robert Okafor",   "age": 78, "bed": "ICU-03", "room": "B1", "base_risk": 88,  "gender": "M"},
    "P004": {"name": "Maria Gonzalez",  "age": 55, "bed": "ICU-04", "room": "B2", "base_risk": 32,  "gender": "F"},
    "P005": {"name": "David Kim",       "age": 82, "bed": "ICU-05", "room": "C1", "base_risk": 78,  "gender": "M"},
    "P006": {"name": "Elena Vasquez",   "age": 38, "bed": "ICU-06", "room": "C2", "base_risk": 20,  "gender": "F"},
}

# Live state
patient_state = {}
# Timeline events per patient (last 20)
patient_timeline = {pid: [] for pid in PATIENTS}

def _init_state(pid):
    info = PATIENTS[pid]
    risk = info["base_risk"]
    return {
        **info,
        "Heart_Rate":        80  + risk * 0.3   + random.gauss(0, 3),
        "Temperature":       37.0 + risk * 0.012 + random.gauss(0, 0.2),
        "Blood_Pressure":    120  - risk * 0.35  + random.gauss(0, 5),
        "Resp_Rate":         16   + risk * 0.07  + random.gauss(0, 1),
        "Oxygen_Level":      99   - risk * 0.08  + random.gauss(0, 0.5),
        "Infection_Marker":  risk / 100          + random.gauss(0, 0.03),
        "risk_score":        float(risk),
        "risk_level":        "High" if risk > 70 else ("Medium" if risk > 30 else "Low"),
        "risk_color":        "#ef4444" if risk > 70 else ("#f59e0b" if risk > 30 else "#10b981"),
        "sirs_score":        0,
        "disclaimer":        "Demo system. Trained on synthetic data. Not validated for clinical use. Not a substitute for clinical judgment.",
        "explanation":       [],
        "contributions":     {"Heart Rate": 20, "Blood Pressure": 20, "Temperature": 20, "SpO2": 20, "Resp Rate": 20},
        "ai_synthesis":      "Initializing clinical AI engine...",
        "alert_level":       "CRITICAL" if risk > 70 else ("WARNING" if risk > 30 else "STABLE"),
        "trend":             [],
        "anomaly":           False,
        "deteriorating":     False,
        "last_updated":      datetime.now().isoformat(),
    }

for pid in PATIENTS:
    patient_state[pid] = _init_state(pid)

# ─────────────────────────────────────────────
# Risk Classification
# ─────────────────────────────────────────────
def get_risk_level(prob):
    if prob < 30:
        return {"level": "Low",    "color": "#10b981", "msg": "Stable. Continue standard monitoring."}
    elif prob < 70:
        return {"level": "Medium", "color": "#f59e0b", "msg": "Elevated risk. Increase monitoring frequency."}
    else:
        return {"level": "High",   "color": "#ef4444", "msg": "CRITICAL: Immediate clinical intervention required."}

# ─────────────────────────────────────────────
# Feature Contributions (SHAP-style approximation)
# ─────────────────────────────────────────────
def compute_contributions(vitals, prob):
    hr_dev   = abs(vitals['Heart_Rate']       - 75)  / 75
    bp_dev   = abs(vitals['Blood_Pressure']   - 110) / 110
    tmp_dev  = abs(vitals['Temperature']      - 37)  / 2
    spo2_dev = abs(vitals['Oxygen_Level']     - 98)  / 10
    rr_dev   = abs(vitals['Resp_Rate']        - 15)  / 15
    inf_dev  = vitals['Infection_Marker']

    total = hr_dev + bp_dev + tmp_dev + spo2_dev + rr_dev + inf_dev + 0.001
    scale = prob / total if total > 0 else 1

    return {
        "Heart Rate":      round(hr_dev   * scale, 1),
        "Blood Pressure":  round(bp_dev   * scale, 1),
        "Temperature":     round(tmp_dev  * scale, 1),
        "SpO2":            round(spo2_dev * scale, 1),
        "Resp Rate":       round(rr_dev   * scale, 1),
        "Infection Mkr":   round(inf_dev  * scale, 1),
    }

# ─────────────────────────────────────────────
# ML Inference Pipeline
# ─────────────────────────────────────────────
def run_ml_pipeline(pid, vitals, generate_ai=False):
    try:
        if model is None:
            return None

        features = [
            vitals['Heart_Rate'], vitals['Temperature'],
            vitals['Blood_Pressure'], vitals['Resp_Rate'],
            vitals['Oxygen_Level'], vitals['Age'],
            vitals['Infection_Marker'],
        ]
        fs    = scaler.transform([features])
        prob  = model.predict_proba(fs)[0][1] * 100
        ri    = get_risk_level(prob)

        explanation = []
        if vitals['Heart_Rate']       > 100:  explanation.append("Tachycardia")
        if vitals['Heart_Rate']       < 60:   explanation.append("Bradycardia")
        if vitals['Temperature']      > 38.0: explanation.append("Fever / Hyperthermia")
        if vitals['Temperature']      < 36.0: explanation.append("Hypothermia")
        if vitals['Blood_Pressure']   < 90:   explanation.append("Hypotension")
        if vitals['Oxygen_Level']     < 94:   explanation.append("Hypoxia")
        if vitals['Resp_Rate']        > 20:   explanation.append("Tachypnea")
        if vitals['Infection_Marker'] > 0.5:  explanation.append("Elevated Infection Marker")

        sirs = sum([
            vitals['Temperature'] < 36 or vitals['Temperature'] > 38,
            vitals['Heart_Rate']  > 90,
            vitals['Resp_Rate']   > 20,
            vitals['Infection_Marker'] > 0.5,
        ])

        contributions = compute_contributions(vitals, prob)
        alert_level   = "CRITICAL" if prob > 70 else ("WARNING" if prob > 30 else "STABLE")

        ai_synthesis = ""
        if generate_ai:
            ai_synthesis = generate_gemini_synthesis(vitals, prob, ri, explanation)

        return {
            "risk_score":    round(prob, 1),
            "risk_level":    ri['level'],
            "risk_color":    ri['color'],
            "message":       ri['msg'],
            "explanation":   explanation or ["All vital signs within normal ranges."],
            "sirs_score":    sirs,
            "contributions": contributions,
            "ai_synthesis":  ai_synthesis,
            "alert_level":   alert_level,
            "disclaimer":    "Demo system. Trained on synthetic data. Not validated for clinical use. Not a substitute for clinical judgment.",
        }
    except Exception as e:
        print(f"[ML Error] {e}")
        return None

# ─────────────────────────────────────────────
# Gemini AI Synthesis
# ─────────────────────────────────────────────
def generate_gemini_synthesis(vitals, prob, ri, explanation):
    try:
        if not GEMINI_API_KEY:
            return _local_synthesis(ri, explanation, prob)
        prompt = (
            f"You are SepsisGuard AI, a clinical decision-support AI in an ICU. "
            f"HR={vitals['Heart_Rate']:.0f}bpm, Temp={vitals['Temperature']:.1f}C, "
            f"SysBP={vitals['Blood_Pressure']:.0f}mmHg, RR={vitals['Resp_Rate']:.0f}bpm, "
            f"SpO2={vitals['Oxygen_Level']:.1f}%, InfMkr={vitals['Infection_Marker']:.2f}. "
            f"Risk={prob:.1f}% ({ri['level']}). Triggers: {', '.join(explanation) or 'None'}. "
            f"Describe the observed physiological pattern and model output. "
            f"Do not issue treatment directives, medication instructions, dosing recommendations, or clinical orders. "
            f"Do not present the model prediction as a diagnosis. "
            f"Frame the output as decision-support information that requires independent clinical verification. "
            f"Do not claim clinical certainty. Write 2 concise clinical observation sentences. No formatting, raw text only."
        )
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        resp = requests.post(url, headers={'Content-Type': 'application/json'},
                             json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=8)
        r = resp.json()
        if "error" in r:
            return _local_synthesis(ri, explanation, prob)
        return r['candidates'][0]['content']['parts'][0]['text'].replace('\n', ' ').strip()
    except Exception:
        return _local_synthesis(ri, explanation, prob)

def _local_synthesis(ri, explanation, prob):
    d = explanation[0] if explanation else "hemodynamic instability"
    if prob > 70:
        return (f"The observed vital-sign pattern is consistent with elevated sepsis risk driven by {d}. "
                f"Independent clinical assessment and correlation with the patient's full clinical context are recommended.")
    elif prob > 30:
        return (f"Moderate sepsis risk pattern detected — {d} observed. "
                f"Increased monitoring frequency and clinical review are recommended.")
    else:
        return (f"Physiological parameters appear stable within standard limits. "
                f"Standard clinical monitoring is indicated.")

# ─────────────────────────────────────────────
# Timeline Event Logger
# ─────────────────────────────────────────────
def log_event(pid, event_type, message):
    ts = datetime.now().strftime("%H:%M:%S")
    evt = {"time": ts, "type": event_type, "msg": message}
    patient_timeline[pid].append(evt)
    if len(patient_timeline[pid]) > 25:
        patient_timeline[pid].pop(0)
    socketio.emit('timeline_event', {"pid": pid, "event": evt})

# ─────────────────────────────────────────────
# Vitals Simulation
# ─────────────────────────────────────────────
def simulate_vitals(pid, tick):
    info      = PATIENTS[pid]
    base_risk = info["base_risk"]
    prev      = patient_state[pid]

    def drift(val, target, noise, lo, hi):
        return max(lo, min(hi, val + (target - val) * 0.05 + random.gauss(0, noise)))

    hr_t  = 75  + base_risk * 0.5   + math.sin(tick * 0.10) * 5
    bp_t  = 120 - base_risk * 0.40  + math.cos(tick * 0.08) * 4
    rr_t  = 14  + base_risk * 0.08  + math.sin(tick * 0.15) * 1.5
    o2_t  = 99  - base_risk * 0.08
    tm_t  = 37.0 + base_risk * 0.015 + math.sin(tick * 0.05) * 0.1
    if_t  = base_risk / 100 * (1 + math.sin(tick * 0.03) * 0.1)

    # Occasional spike for critical patients
    if base_risk > 60 and random.random() < 0.04:
        hr_t += random.choice([-12, 18])
        o2_t -= random.uniform(1.5, 4)

    return {
        "Heart_Rate":       round(drift(prev["Heart_Rate"],       hr_t,  1.8, 30,  200), 1),
        "Temperature":      round(drift(prev["Temperature"],      tm_t,  0.04, 34, 42),  2),
        "Blood_Pressure":   round(drift(prev["Blood_Pressure"],   bp_t,  1.8, 50,  200), 1),
        "Resp_Rate":        round(drift(prev["Resp_Rate"],        rr_t,  0.7, 6,   45),  1),
        "Oxygen_Level":     round(drift(prev["Oxygen_Level"],     o2_t,  0.25, 70, 100), 1),
        "Infection_Marker": round(max(0, min(1, drift(prev["Infection_Marker"], if_t, 0.008, 0, 1))), 3),
        "Age": info["age"],
    }

# ─────────────────────────────────────────────
# Telemetry Loop
# ─────────────────────────────────────────────
telemetry_running = False
_prev_alert = {pid: None for pid in PATIENTS}

def telemetry_loop():
    global telemetry_running
    tick       = 0
    ai_counter = {pid: random.randint(0, 10) for pid in PATIENTS}
    print("[OK] Telemetry Engine started")

    while telemetry_running:
        try:
            for pid in PATIENTS:
                vitals    = simulate_vitals(pid, tick)
                patient_state[pid].update(vitals)

                gen_ai = (ai_counter[pid] == 0)
                result = run_ml_pipeline(pid, vitals, generate_ai=gen_ai)

                if result:
                    prev_risk  = patient_state[pid].get("risk_score", 0)
                    prev_alert = _prev_alert[pid]
                    new_alert  = result["alert_level"]

                    patient_state[pid].update(result)
                    patient_state[pid]["last_updated"] = datetime.now().isoformat()

                    # Trend
                    trend = patient_state[pid].get("trend", [])
                    trend.append(round(result["risk_score"], 1))
                    if len(trend) > 40:
                        trend.pop(0)
                    patient_state[pid]["trend"] = trend

                    # Deterioration flag
                    delta = result["risk_score"] - prev_risk
                    patient_state[pid]["deteriorating"] = delta > 3

                    # Anomaly flag (sudden spike)
                    patient_state[pid]["anomaly"] = abs(delta) > 8

                    # Timeline events
                    if new_alert == "CRITICAL" and prev_alert != "CRITICAL":
                        log_event(pid, "CRITICAL", f"Sepsis risk escalated to {result['risk_score']:.0f}%")
                    elif new_alert == "WARNING" and prev_alert == "STABLE":
                        log_event(pid, "WARNING", f"Risk rising to {result['risk_score']:.0f}%")
                    elif new_alert == "STABLE" and prev_alert in ("WARNING", "CRITICAL"):
                        log_event(pid, "STABLE", "Patient stabilizing")

                    for trigger in result["explanation"][:2]:
                        log_event(pid, "TRIGGER", trigger)

                    _prev_alert[pid] = new_alert

                ai_counter[pid] = (ai_counter[pid] + 1) % 22

                packet = {
                    "pid":           pid,
                    "vitals":        vitals,
                    "risk_score":    patient_state[pid].get("risk_score",    0),
                    "risk_level":    patient_state[pid].get("risk_level",    "Low"),
                    "risk_color":    patient_state[pid].get("risk_color",    "#10b981"),
                    "alert_level":   patient_state[pid].get("alert_level",   "STABLE"),
                    "sirs_score":    patient_state[pid].get("sirs_score",    0),
                    "disclaimer":    patient_state[pid].get("disclaimer",    "Demo system. Trained on synthetic data. Not validated for clinical use. Not a substitute for clinical judgment."),
                    "explanation":   patient_state[pid].get("explanation",   []),
                    "contributions": patient_state[pid].get("contributions", {}),
                    "ai_synthesis":  patient_state[pid].get("ai_synthesis",  ""),
                    "trend":         patient_state[pid].get("trend",         []),
                    "deteriorating": patient_state[pid].get("deteriorating", False),
                    "anomaly":       patient_state[pid].get("anomaly",       False),
                    "timestamp":     datetime.now().isoformat(),
                    "name":          PATIENTS[pid]["name"],
                    "bed":           PATIENTS[pid]["bed"],
                    "room":          PATIENTS[pid]["room"],
                    "age":           PATIENTS[pid]["age"],
                    "gender":        PATIENTS[pid]["gender"],
                }
                socketio.emit('telemetry', sanitize(packet))

        except Exception as e:
            print(f"[Telemetry Error] {e}")

        tick += 1
        time.sleep(1.2)

# ─────────────────────────────────────────────
# Socket.IO Events
# ─────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    global telemetry_running
    print(f"[+] Dashboard connected: {request.sid}")
    # Full snapshot + timeline
    emit('snapshot', sanitize({pid: patient_state[pid] for pid in PATIENTS}))
    emit('timeline_snapshot', sanitize({pid: patient_timeline[pid] for pid in PATIENTS}))
    if not telemetry_running:
        telemetry_running = True
        threading.Thread(target=telemetry_loop, daemon=True).start()

@socketio.on('disconnect')
def on_disconnect():
    print(f"[-] Dashboard disconnected: {request.sid}")

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
            emit('ai_synthesis_result', sanitize({"pid": pid, **result}))

@socketio.on('copilot_query')
def on_copilot(data):
    """AI Copilot — answer doctor questions about ICU patients."""
    question = data.get('question', '')
    pid      = data.get('pid')

    # Build context from all patients or specific patient
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
        # All patients summary
        lines = []
        for p, ps in patient_state.items():
            lines.append(f"{ps['name']} ({ps['bed']}): risk {ps.get('risk_score',0):.0f}%, {ps.get('alert_level','?')}")
        context = "ICU Overview: " + " | ".join(lines)

    answer = _copilot_answer(question, context)
    emit('copilot_response', {"answer": answer, "pid": pid})

def _copilot_answer(question, context):
    try:
        if GEMINI_API_KEY:
            prompt = (
                f"You are SepsisGuard Copilot, an ICU clinical decision-support AI. "
                f"Context: {context}. "
                f"Doctor asks: '{question}'. "
                f"Describe observed patterns and information from the context. "
                f"Do not issue treatment directives, medication instructions, dosing recommendations, or clinical orders. "
                f"Frame responses as decision-support requiring independent clinical verification. "
                f"Answer concisely in 2-3 sentences, clinically precise, no bullet points."
            )
            url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            resp = requests.post(url, headers={'Content-Type': 'application/json'},
                                 json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=8)
            r = resp.json()
            if "error" not in r:
                return r['candidates'][0]['content']['parts'][0]['text'].replace('\n', ' ').strip()
    except Exception:
        pass
    # Local fallback
    q = question.lower()
    if "risk" in q or "why" in q:
        return f"Based on current telemetry: {context[:200]}. Review observed clinical indicators for further assessment."
    if "intervention" in q or "first" in q or "priority" in q:
        critical = [ps['name'] for ps in patient_state.values() if ps.get('alert_level') == 'CRITICAL']
        return f"Patients currently presenting high risk indicators: {', '.join(critical) if critical else 'None at this time'}. Independent clinical evaluation required."
    return f"Current ICU status summary: {context[:300]}. Consult patient detail cards for vital sign trends."

@app.route('/')
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, 'dashboard.html')

@app.route('/patient')
def serve_patient():
    return send_from_directory(FRONTEND_DIR, 'patient.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "SepsisGuard v3.0 Online", "patients": len(PATIENTS)})

@app.route('/patients', methods=['GET'])
def get_patients():
    return jsonify(sanitize(patient_state))

@app.route('/<path:filename>')
def serve_static(filename):
    if filename.startswith('socket.io'):
        from flask import abort
        abort(404)
    return send_from_directory(FRONTEND_DIR, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data   = request.json
        result = run_ml_pipeline("manual", data, generate_ai=data.get('generate_synthesis', False))
        if result:
            return jsonify(sanitize(result)), 200
        return jsonify({"error": "Inference failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  SepsisGuard AI v3.0 - ICU Intelligence Ecosystem")
    print("  http://localhost:5000")
    print("=" * 55)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
