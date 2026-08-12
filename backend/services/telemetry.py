# -*- coding: utf-8 -*-
"""
Telemetry Simulation & Patient Registry Service (Phase 10 / Phase 12)
Manages in-memory ICU patient registry, vital sign simulation engine, and event logging using
structured logging.

IMPORTANT NOTICE:
All patient records, vital sign streams, and telemetry loops in this module are
SIMULATED DEMO DATA for system demonstration and prototyping purposes only.
"""

import math
import time
import random
import threading
from datetime import datetime
from backend.config import logger
from backend.validation.schemas import sanitize
from backend.ml.inference import run_ml_pipeline

PATIENTS = {
    "P001": {"name": "James Hartwell",  "age": 67, "bed": "ICU-01", "room": "A1", "base_risk": 15,  "gender": "M"},
    "P002": {"name": "Sarah Chen",      "age": 45, "bed": "ICU-02", "room": "A2", "base_risk": 72,  "gender": "F"},
    "P003": {"name": "Robert Okafor",   "age": 78, "bed": "ICU-03", "room": "B1", "base_risk": 88,  "gender": "M"},
    "P004": {"name": "Maria Gonzalez",  "age": 55, "bed": "ICU-04", "room": "B2", "base_risk": 32,  "gender": "F"},
    "P005": {"name": "David Kim",       "age": 82, "bed": "ICU-05", "room": "C1", "base_risk": 78,  "gender": "M"},
    "P006": {"name": "Elena Vasquez",   "age": 38, "bed": "ICU-06", "room": "C2", "base_risk": 20,  "gender": "F"},
}

patient_state = {}
patient_timeline = {pid: [] for pid in PATIENTS}
telemetry_running = False
_prev_alert = {pid: None for pid in PATIENTS}

def _init_state(pid):
    info = PATIENTS[pid]
    risk = info["base_risk"]
    vitals = {
        "Heart_Rate":        round(80  + risk * 0.38   + random.gauss(0, 2), 1),
        "Temperature":       round(37.0 + risk * 0.018 + random.gauss(0, 0.1), 2),
        "Blood_Pressure":    round(120  - risk * 0.40  + random.gauss(0, 3), 1),
        "Resp_Rate":         round(16   + risk * 0.09  + random.gauss(0, 1), 1),
        "Oxygen_Level":      round(99   - risk * 0.09  + random.gauss(0, 0.5), 1),
        "Infection_Marker":  round(max(0, min(1, risk / 100 + random.gauss(0, 0.02))), 3),
        "Age": info["age"],
    }
    
    ml_res = run_ml_pipeline(pid, vitals, generate_ai=True)
    if not ml_res:
        ml_res = {
            "risk_score": float(risk),
            "risk_level": "High" if risk >= 50 else ("Medium" if risk >= 27 else "Low"),
            "risk_color": "#ef4444" if risk >= 50 else ("#f59e0b" if risk >= 27 else "#10b981"),
            "alert_level": "CRITICAL" if risk >= 50 else ("WARNING" if risk >= 27 else "STABLE"),
            "sirs_score": 0,
            "explanation": [],
            "contributions": {},
            "shap_explanation": {},
            "ai_synthesis": "Initializing clinical AI engine..."
        }

    return {
        **info,
        **vitals,
        **ml_res,
        "trend": [round(max(0, min(100, ml_res["risk_score"] + math.sin(i * 0.3) * 3 + random.gauss(0, 1.5))), 1) for i in range(20)],
        "anomaly": False,
        "deteriorating": False,
        "last_updated": datetime.now().isoformat(),
    }

for pid in PATIENTS:
    patient_state[pid] = _init_state(pid)

def log_event(socketio_instance, pid, event_type, message):
    ts = datetime.now().strftime("%H:%M:%S")
    evt = {"time": ts, "type": event_type, "msg": message}
    patient_timeline[pid].append(evt)
    if len(patient_timeline[pid]) > 25:
        patient_timeline[pid].pop(0)
    if socketio_instance:
        socketio_instance.emit('timeline_event', {"pid": pid, "event": evt}, room='icu_unit_a')

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

def telemetry_loop(socketio_instance):
    global telemetry_running
    tick       = 0
    ai_counter = {pid: random.randint(0, 10) for pid in PATIENTS}
    logger.info("Telemetry Engine simulation loop started")

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

                    trend = patient_state[pid].get("trend", [])
                    trend.append(round(result["risk_score"], 1))
                    if len(trend) > 40:
                        trend.pop(0)
                    patient_state[pid]["trend"] = trend

                    delta = result["risk_score"] - prev_risk
                    patient_state[pid]["deteriorating"] = delta > 3
                    patient_state[pid]["anomaly"] = abs(delta) > 8

                    if new_alert == "CRITICAL" and prev_alert != "CRITICAL":
                        log_event(socketio_instance, pid, "CRITICAL", f"Sepsis risk escalated to {result['risk_score']:.0f}%")
                    elif new_alert == "WARNING" and prev_alert == "STABLE":
                        log_event(socketio_instance, pid, "WARNING", f"Risk rising to {result['risk_score']:.0f}%")
                    elif new_alert == "STABLE" and prev_alert in ("WARNING", "CRITICAL"):
                        log_event(socketio_instance, pid, "STABLE", "Patient stabilizing")

                    for trigger in result["explanation"][:2]:
                        log_event(socketio_instance, pid, "TRIGGER", trigger)

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
                    "shap_explanation": patient_state[pid].get("shap_explanation", {}),
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
                if socketio_instance:
                    socketio_instance.emit('telemetry', sanitize(packet), room='icu_unit_a')

        except Exception as e:
            logger.error(f"Telemetry simulation loop exception: {e}")

        tick += 1
        time.sleep(1.2)

def start_telemetry_thread(socketio_instance):
    global telemetry_running
    if not telemetry_running:
        telemetry_running = True
        threading.Thread(target=telemetry_loop, args=(socketio_instance,), daemon=True).start()
