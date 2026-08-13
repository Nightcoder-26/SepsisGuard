# -*- coding: utf-8 -*-
"""
AI Copilot Service Module (Phase 10 / Phase 12)
Local narrative synthesis only — no external API calls during telemetry loop.
Gemini API is retained only for the interactive Copilot chat (on-demand).
Enforces non-clinical safety restrictions: no treatment directives or orders.
"""

from backend.config import GEMINI_API_KEY, logger


def generate_gemini_synthesis(vitals, prob, ri, explanation, shap_explanation=None):
    """
    Generates a concise clinical observation narrative grounded in SHAP attributions.
    Uses local template synthesis — no Gemini API call during telemetry loop.
    This prevents excessive external calls and eliminates rate-limit warnings.
    """
    return _local_synthesis(ri, explanation, prob, shap_explanation)


def _local_synthesis(ri, explanation, prob, shap_explanation=None):
    top_feature = "key physiological indicators"
    second_feature = None

    if isinstance(shap_explanation, dict) and "features" in shap_explanation and shap_explanation["features"]:
        risk_features = [f for f in shap_explanation["features"] if f.get("direction") == "increases_risk"]
        if risk_features:
            top_feature = risk_features[0].get("display_name", "key vitals")
            if len(risk_features) > 1:
                second_feature = risk_features[1].get("display_name")
        else:
            top_feature = shap_explanation["features"][0].get("display_name", "key vitals")
    elif explanation:
        top_feature = explanation[0]
        if len(explanation) > 1:
            second_feature = explanation[1]

    factor_str = top_feature
    if second_feature:
        factor_str = f"{top_feature} and {second_feature}"

    if prob >= 50:
        return (
            f"The model estimates an elevated sepsis risk ({prob:.0f}%) primarily attributed to {factor_str}. "
            f"Independent clinical assessment and correlation with the patient's full context are recommended."
        )
    elif prob >= 27:
        return (
            f"The model indicates moderate sepsis risk ({prob:.0f}%) above the operating threshold (0.27), "
            f"driven by model attribution from {factor_str}. "
            f"Increased monitoring frequency and clinical review are indicated."
        )
    else:
        return (
            f"The model estimates a low risk pattern ({prob:.0f}%) with contributing factors within standard reference ranges. "
            f"Standard clinical monitoring is indicated."
        )


def copilot_answer(question, context):
    """
    Answers clinician questions about ICU patients.
    Attempts Gemini API for interactive on-demand queries; falls back to local rules.
    This is only called on explicit clinician interaction, not on the telemetry loop.
    """
    try:
        if GEMINI_API_KEY:
            import requests
            prompt = (
                f"You are SepsisGuard Copilot, an ICU clinical decision-support AI. "
                f"Context: {context}. "
                f"Doctor asks: '{question}'. "
                f"Describe observed patterns and information from the context. "
                f"Do not issue treatment directives, medication instructions, dosing recommendations, or clinical orders. "
                f"Frame responses as decision-support requiring independent clinical verification. "
                f"Answer concisely in 2-3 sentences, clinically precise, no bullet points."
            )
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            resp = requests.post(
                url,
                headers={'Content-Type': 'application/json'},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=8
            )
            r = resp.json()
            if "error" not in r:
                return r['candidates'][0]['content']['parts'][0]['text'].replace('\n', ' ').strip()
            else:
                logger.warning("Gemini Copilot API error; falling back to local rules.")
    except Exception as e:
        logger.warning(f"Gemini Copilot request exception ({e}); falling back to local rules.")

    q = question.lower()
    if "risk" in q or "why" in q:
        return f"Based on current telemetry: {context[:200]}. Review observed clinical indicators for further assessment."
    if "intervention" in q or "first" in q or "priority" in q:
        return f"Current ICU telemetry status summary: {context[:250]}. Independent clinical evaluation required."
    return f"Current ICU status summary: {context[:300]}. Consult patient detail cards for vital sign trends."
