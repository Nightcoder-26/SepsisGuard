# -*- coding: utf-8 -*-
"""
AI Copilot Service Module (Phase 10 / Phase 12)
Handles Gemini LLM prompt generation, API calls, and local narrative synthesis.
Enforces non-clinical safety restrictions: no treatment directives or orders. Uses structured logging.
"""

import requests
from backend.config import GEMINI_API_KEY, logger

def generate_gemini_synthesis(vitals, prob, ri, explanation, shap_explanation=None):
    """
    Generates a concise 2-sentence clinical observation narrative from Gemini API grounded in SHAP attributions
    or falls back to local template synthesis if offline.
    """
    try:
        if not GEMINI_API_KEY:
            logger.info("Gemini API key not configured; using local narrative synthesis fallback.")
            return _local_synthesis(ri, explanation, prob, shap_explanation)
            
        shap_factors = []
        if isinstance(shap_explanation, dict) and "features" in shap_explanation:
            for f in shap_explanation["features"][:3]:
                shap_factors.append(f"{f.get('display_name', f.get('feature'))} ({f.get('direction', 'contributes')})")
        shap_str = ", ".join(shap_factors) if shap_factors else ", ".join(explanation[:3])

        prompt = (
            f"You are explaining an ML model risk output to a healthcare professional in an ICU. "
            f"Model Sepsis Risk Estimate: {prob:.1f}% ({ri['level']}). "
            f"Top SHAP Model-Attributed Features: {shap_str}. "
            f"Current Vitals: HR={vitals['Heart_Rate']:.0f}bpm, Temp={vitals['Temperature']:.1f}C, "
            f"SysBP={vitals['Blood_Pressure']:.0f}mmHg, RR={vitals['Resp_Rate']:.0f}bpm, SpO2={vitals['Oxygen_Level']:.1f}%. "
            f"Instructions: "
            f"1. Explain why the model produced its risk estimate based ONLY on the supplied SHAP model evidence. "
            f"2. Do NOT diagnose sepsis. Do NOT claim clinical certainty. "
            f"3. Do NOT prescribe medications, antibiotics, fluids, vasopressors, or clinical treatment directives. "
            f"4. Do NOT invent missing patient information. "
            f"5. Frame the output as model explanation for clinician decision-support requiring independent clinical judgment. "
            f"Write exactly 2 concise clinical observation sentences. No formatting, raw text only."
        )
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        resp = requests.post(url, headers={'Content-Type': 'application/json'},
                             json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=8)
        r = resp.json()
        if "error" in r:
            logger.warning("Gemini API returned error response; falling back to local synthesis.")
            return _local_synthesis(ri, explanation, prob, shap_explanation)
        return r['candidates'][0]['content']['parts'][0]['text'].replace('\n', ' ').strip()
    except Exception as e:
        logger.warning(f"Gemini API request exception ({e}); falling back to local synthesis.")
        return _local_synthesis(ri, explanation, prob, shap_explanation)

def _local_synthesis(ri, explanation, prob, shap_explanation=None):
    top_feature = "key physiological indicators"
    if isinstance(shap_explanation, dict) and "features" in shap_explanation and shap_explanation["features"]:
        top_feature = shap_explanation["features"][0].get("display_name", "key vitals")
    elif explanation:
        top_feature = explanation[0]

    if prob >= 50:
        return (f"The model estimates an elevated sepsis risk ({prob:.0f}%) influenced primarily by model attributions from {top_feature}. "
                f"Independent clinical assessment and correlation with the patient's full context are recommended.")
    elif prob >= 27:
        return (f"The model indicates moderate sepsis risk ({prob:.0f}%) above operating threshold (0.27), driven by {top_feature}. "
                f"Increased monitoring frequency and clinical review are recommended.")
    else:
        return (f"The model estimates a low risk pattern ({prob:.0f}%) with physiological parameters within standard reference ranges. "
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
