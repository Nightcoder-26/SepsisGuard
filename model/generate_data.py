import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

def generate_sepsis_data(n_samples=2000):
    """
    Generates a synthetic sepsis dataset based on clinical patterns.
    High HR, High RespRate, Low BP, and Abnormal Temp are markers of Sepsis.
    """
    
    # Base clinical normal values
    # Heart Rate (BPM): 60-100
    # Temperature (C): 36-37.5
    # BP Systolic (mmHg): 90-120
    # Resp Rate (BPM): 12-20
    # SpO2 (%): 95-100
    
    data = []
    
    for _ in range(n_samples):
        # Determine if this sample will be a "Sepsis case" (approx 30% chance for imbalance)
        is_sepsis = np.random.choice([0, 1], p=[0.7, 0.3])
        
        age = np.random.randint(18, 90)
        infection_marker = np.random.uniform(0, 1) # Probability of having a primary infection
        
        if is_sepsis:
            # Sepsis pattern: High HR, Fever/Hypothermia, High RR, Low BP, Low SpO2
            hr = np.random.normal(110, 15)
            temp = np.random.choice([np.random.normal(38.5, 0.8), np.random.normal(35.5, 0.8)])
            bp_sys = np.random.normal(85, 10)
            resp_rate = np.random.normal(24, 4)
            spo2 = np.random.normal(92, 3)
            # Ensure physiological limits
            hr = max(40, min(200, hr))
            temp = max(34, min(41, temp))
            bp_sys = max(50, min(180, bp_sys))
            resp_rate = max(8, min(40, resp_rate))
            spo2 = max(70, min(100, spo2))
            infection_marker = min(1.0, infection_marker + 0.5)
        else:
            # Normal pattern: Stable vitals
            hr = np.random.normal(75, 10)
            temp = np.random.normal(36.8, 0.5)
            bp_sys = np.random.normal(115, 15)
            resp_rate = np.random.normal(16, 2)
            spo2 = np.random.normal(98, 1)
            # Ensure physical limits
            hr = max(50, min(120, hr))
            temp = max(36, min(37.8, temp))
            bp_sys = max(90, min(160, bp_sys))
            resp_rate = max(10, min(24, resp_rate))
            spo2 = max(94, min(100, spo2))
        
        data.append({
            'Heart_Rate': round(hr, 1),
            'Temperature': round(temp, 1),
            'Blood_Pressure': round(bp_sys, 1),
            'Resp_Rate': round(resp_rate, 1),
            'Oxygen_Level': round(spo2, 1),
            'Age': age,
            'Infection_Marker': round(infection_marker, 2),
            'Sepsis_Risk': is_sepsis
        })
        
    df = pd.DataFrame(data)
    
    # Save to CSV
    dataset_path = os.path.join('dataset', 'sepsis_data.csv')
    df.to_csv(dataset_path, index=False)
    print(f"✅ Dataset generated successfully with {n_samples} samples.")
    print(f"📁 Saved to: {dataset_path}")

if __name__ == "__main__":
    generate_sepsis_data()
