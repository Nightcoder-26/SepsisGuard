import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

def train_sepsis_model():
    # 1. Load Data
    data_path = os.path.join('dataset', 'sepsis_data.csv')
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    X = df.drop('Sepsis_Risk', axis=1)
    y = df['Sepsis_Risk']
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Preprocessing (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Models
    # --- Random Forest ---
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    
    # --- Logistic Regression ---
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    
    print("-" * 30)
    print("MODEL COMPARISON")
    print(f"Random Forest - Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")
    print(f"Logistic Reg  - Accuracy: {lr_acc:.4f}, F1: {lr_f1:.4f}")
    print("-" * 30)
    
    # 5. Export Feature Importance
    importances = rf_model.feature_importances_
    features = X.columns
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance_df)
    
    # 6. Save Model and Scaler to Backend directory for deployment
    os.makedirs('backend', exist_ok=True)
    joblib.dump(rf_model, os.path.join('backend', 'model.joblib'))
    joblib.dump(scaler, os.path.join('backend', 'scaler.joblib'))
    
    # Save a metadata file for the frontend to use
    metadata = {
        'accuracy': round(rf_acc, 4),
        'features': list(features),
        'importance': feature_importance_df.to_dict(orient='records')
    }
    joblib.dump(metadata, os.path.join('backend', 'metadata.joblib'))
    
    print("\nModel saved successfully in 'backend/' folder.")

    # 7. Visualizations (Saved for Presentation)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Sepsis Detection - Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, rf_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    print("Plots saved as feature_importance.png and confusion_matrix.png")

if __name__ == "__main__":
    train_sepsis_model()
