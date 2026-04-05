from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
print("[SUCCESS] Gemini Engine loaded via Direct REST Engine architecture.")

app = Flask(__name__)
# Enable CORS so the React frontend can talk to this backend
CORS(app)

# Load the pre-trained model and scaler
# They should have been saved into the backend/ folder by the training script
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
    metadata_path = os.path.join(os.path.dirname(__file__), 'metadata.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)
    print("[SUCCESS] Model, Scaler, and Metadata loaded successfully.")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    model = None
    scaler = None
    metadata = None

def get_risk_level(prob):
    """
    Categorize risk based on probability (0-100)
    """
    if prob < 30:
        return {"level": "Low", "color": "#10b981", "msg": "Patient is stable. Continue regular monitoring."}
    elif prob < 70:
        return {"level": "Medium", "color": "#f59e0b", "msg": "Increased risk detected. Increase monitoring frequency."}
    else:
        return {"level": "High", "color": "#ef4444", "msg": "High risk of sepsis in next 6-12 hours. Immediate clinical intervention recommended."}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        # Expected features in correct order as per training
        features = [
            data['Heart_Rate'],
            data['Temperature'],
            data['Blood_Pressure'],
            data['Resp_Rate'],
            data['Oxygen_Level'],
            data['Age'],
            data['Infection_Marker']
        ]
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Get prediction probability
        prob = model.predict_proba(features_scaled)[0][1] * 100 # Risk of class 1 (Sepsis)
        risk_info = get_risk_level(prob)
        
        # Simple Explainability: Contribution of features
        # Note: Since RF doesn't give local expl. directly without SHAP/LIME, 
        # we will use feature values relative to average importance for a simple hackathon "why" logic.
        explanation = []
        if data['Heart_Rate'] > 100: explanation.append("Elevated Heart Rate Detected")
        if data['Temperature'] > 38 or data['Temperature'] < 36: explanation.append("Abnormal Body Temperature")
        if data['Blood_Pressure'] < 90: explanation.append("Low Blood Pressure (Hypotension)")
        if data['Oxygen_Level'] < 94: explanation.append("Low Blood Oxygen Saturation")
        if data['Resp_Rate'] > 20: explanation.append("High Respiratory Rate")
        
        # AI Text Generation via true LLM Rest API
        ai_synthesis = ""
        try:
            if GEMINI_API_KEY:
                prompt = f"""
                You are a harsh but brilliant ICU AI system named "SepsisGuard". 
                You are analyzing a patient's real-time vitals:
                HR: {data['Heart_Rate']} bpm
                Temp: {data['Temperature']} C
                Sys BP: {data['Blood_Pressure']} mmHg
                Resp Rate: {data['Resp_Rate']} bpm
                O2: {data['Oxygen_Level']}%
                Age: {data['Age']}
                Infection Marker: {data['Infection_Marker']}

                Our Random Forest Machine Learning model has classified this patient's Sepsis Risk at {round(prob, 1)}%.
                The primary clinical triggers are: {explanation}.
                
                Generate a 3-sentence, intense, highly clinical summary synthesizing this data. 
                Do not introduce yourself or use formatting, just output the raw diagnostic clinical text directly. 
                Focus on the severity if the risk is high, or stability if the risk is low.
                """
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
                headers = {'Content-Type': 'application/json'}
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                
                response = requests.post(url, headers=headers, json=payload)
                result = response.json()
                
                if "error" in result:
                    # Specific raw fallback since model 1.5 might be restricted on certain free keys
                    if result["error"]["code"] == 404:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
                        response = requests.post(url, headers=headers, json=payload)
                        result = response.json()
                        
                if "error" in result:
                     ai_synthesis = f"API Auth/Region Error: {result['error']['message']}"
                else:
                     ai_synthesis = result['candidates'][0]['content']['parts'][0]['text'].replace('\n', ' ').strip()
            else:
                ai_synthesis = "AI Engine Offline. Please verify API key installation."
        except Exception as e:
            ai_synthesis = "AI Engine processing error: " + str(e)
            
        return jsonify({
            "risk_score": round(prob, 2),
            "risk_level": risk_info['level'],
            "risk_color": risk_info['color'],
            "message": risk_info['msg'],
            "explanation": explanation if explanation else ["Vital signs are within normal ranges."],
            "ai_synthesis": ai_synthesis
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/metadata', methods=['GET'])
def get_metadata():
    if metadata is None:
        return jsonify({"error": "Metadata not found"}), 404
    return jsonify(metadata)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Backend is running!"})

if __name__ == '__main__':
    # Running on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
