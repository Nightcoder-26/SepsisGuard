# SepsisGuard AI - Early Warning Detection System 🏥🤖

SepsisGuard AI is a predictive healthcare solution that leverages Machine Learning to identify the early onset of sepsis 6–12 hours before clinical diagnosis. Early detection of sepsis is critical for patient survival, as mortality risk increases by 8% for every hour treatment is delayed.

## 🚀 Key Features
- **Early Risk Prediction**: Machine Learning model (Random Forest) that predicts sepsis risk based on vital signs.
- **Explainable AI (XAI)**: Provides clinical reasoning for every risk score (e.g., "Elevated Heart Rate + Low Blood Pressure").
- **Dynamic Risk Gauge**: Interactive, color-coded gauge (Low, Medium, High Risk).
- **Trend Visualization**: Real-time tracking of risk levels using Chart.js.
- **Crisis Simulation**: One-click "Simulate Crisis" mode for clinical demonstrations.
- **Modern UI**: Dark-mode, glassmorphic dashboard built for high-stress ICU environments.

## 📁 Project Structure
```text
├── backend/
│   ├── app.py             # Flask API logic
│   ├── model.joblib       # Trained Random Forest Model
│   ├── scaler.joblib      # Data Preprocessing Scaler
│   └── metadata.joblib    # Model stats & feature importance
├── frontend/
│   ├── index.html         # Main dashboard structure
│   ├── style.css          # Premium glassmorphic styling
│   └── script.js          # Interactive logic & data binding
├── model/
│   ├── generate_data.py   # SIRS-based synthetic data generator
│   └── train_model.py     # Model training & comparison script
├── dataset/
│   └── sepsis_data.csv    # Generated clinical dataset
└── README.md              # Documentation
```

## 🛠️ How to Run Locally

### 1. Prerequisite (Python)
Ensure Python 3.8+ is installed.

### 2. Setup Backend
Open your terminal and navigate to the project root:
```bash
# Install dependencies
pip install flask flask-cors scikit-learn pandas numpy joblib

# Run the Flask server
cd backend
python app.py
```
The backend will run on `http://localhost:5000`.

### 3. Run Frontend
1. Navigate to the `frontend/` folder.
2. Open `index.html` in any modern web browser (Chrome, Edge, etc.).
   *(No web server required, but ensure the backend is running for live predictions)*

## 🧠 Model Information
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~98-100% (on SIRS-based synthetic data)
- **Features**: Heart Rate, Temperature, BP, Respiratory Rate, SpO2, Age, Infection Markers (WBC/CRP).

## 🏆 Development Team
Build with 💡 for the **Sushack 4.0 Hackathon**.
