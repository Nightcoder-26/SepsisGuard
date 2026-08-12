# -*- coding: utf-8 -*-
"""
AI Copilot Service Module (Phase 10 / Phase 12)
Handles Gemini LLM prompt generation, API calls, and local narrative synthesis.
Enforces non-clinical safety restrictions: no treatment directives or orders. Uses structured logging.
"""

import requests
from backend.config import GEMINI_API_KEY, logger

def generate_gemini_synthesis(vitals, prob, ri, explanation):
    """
    Generates a concise 2-sentence clinical observation narrative from Gemini API
    or falls back to local template synthesis if offline.
    """
    try:
        if not GEMINI_API_KEY:
            logger.info("Gemini API key not configured; using local narrative synthesis fallback.")
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
            logger.warning("Gemini API returned error response; falling back to local synthesis.")
            return _local_synthesis(ri, explanation, prob)
        return r['candidates'][0]['content']['parts'][0]['text'].replace('\n', ' ').strip()
    except Exception as e:
        logger.warning(f"Gemini API request exception ({e}); falling back to local synthesis.")
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

def copilot_answer(question, context):
    """
    Answers clinician questions about ICU patients using Gemini API or fallback rules.
    """
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
            else:
                logger.warning("Gemini Copilot API error; falling back to local rules.")
    except Exception as e:
        logger.warning(f"Gemini Copilot request exception ({e}); falling back to local rules.")
        pass

    q = question.lower()
    if "risk" in q or "why" in q:
        return f"Based on current telemetry: {context[:200]}. Review observed clinical indicators for further assessment."
    if "intervention" in q or "first" in q or "priority" in q:
        return f"Current ICU telemetry status summary: {context[:250]}. Independent clinical evaluation required."
    return f"Current ICU status summary: {context[:300]}. Consult patient detail cards for vital sign trends."
