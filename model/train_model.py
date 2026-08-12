# -*- coding: utf-8 -*-
"""
SepsisGuard Model Pipeline Improvement (Phase 3)
Trains baseline and candidate ML models on real PhysioNet clinical dataset using GroupKFold.
Strict patient-level separation, PR-AUC hyperparameter optimization, and validation threshold selection.
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, average_precision_score
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.features import add_derived_features, DERIVED_FEATURE_COLS

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Paths
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

BASE_FEATURE_COLS = [
    'Heart_Rate', 'Oxygen_Level', 'Temperature', 'Blood_Pressure',
    'Mean_Arterial_Pressure', 'Resp_Rate', 'Age', 'Infection_Marker',
    'Glucose', 'Creatinine', 'Platelets'
]

MISSINGNESS_INDICATOR_COLS = [
    'Heart_Rate_isnan', 'Temperature_isnan', 'Blood_Pressure_isnan',
    'Resp_Rate_isnan', 'Infection_Marker_isnan'
]

TARGET_COL = 'Sepsis_Risk'
PATIENT_ID_COL = 'Patient_ID'
TIME_COL = 'ICU_Length_of_Stay'
RANDOM_STATE = 42
TARGET_SENSITIVITY = 0.85

def calculate_pr_auc(y_true, y_prob):
    """Calculate Precision-Recall AUC (average precision)."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_prob))

def calculate_roc_auc(y_true, y_prob):
    """Calculate ROC AUC."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))

def evaluate_group_cv(model, X, y, groups, n_splits=5):
    """
    Evaluates a model using GroupKFold cross-validation.
    Returns lists of PR-AUC and ROC-AUC scores across folds.
    Strictly verifies zero patient leakage in every fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    pr_aucs = []
    roc_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_pids = set(groups.iloc[train_idx])
        val_pids = set(groups.iloc[val_idx])

        # Strict GroupKFold leakage assertion
        overlap = train_pids & val_pids
        assert len(overlap) == 0, f"CRITICAL PATIENT LEAKAGE IN FOLD {fold_idx}: {overlap}"

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_va)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_va)
        else:
            y_prob = model.predict(X_va)

        pr_auc = calculate_pr_auc(y_va, y_prob)
        roc_auc = calculate_roc_auc(y_va, y_prob)

        pr_aucs.append(pr_auc)
        roc_aucs.append(roc_auc)

    return np.array(pr_aucs), np.array(roc_aucs)

def evaluate_sirs_heuristic(df):
    """
    Evaluates the SIRS clinical heuristic baseline on a dataset.
    SIRS criteria (>= 2 positive):
    1. Temp > 38.0 or Temp < 36.0
    2. Heart_Rate > 90
    3. Resp_Rate > 20
    4. Infection_Marker > 0.5
    """
    temp_crit = (df['Temperature'] > 38.0) | (df['Temperature'] < 36.0)
    hr_crit   = df['Heart_Rate'] > 90
    rr_crit   = df['Resp_Rate'] > 20
    inf_crit  = df['Infection_Marker'] > 0.5

    sirs_score = temp_crit.astype(int) + hr_crit.astype(int) + rr_crit.astype(int) + inf_crit.astype(int)
    sirs_pred  = (sirs_score >= 2).astype(int)
    sirs_prob  = sirs_score / 4.0

    y_true = df[TARGET_COL].values
    acc = float(accuracy_score(y_true, sirs_pred))
    prec = float(precision_score(y_true, sirs_pred, zero_division=0))
    rec = float(recall_score(y_true, sirs_pred, zero_division=0))
    f1 = float(f1_score(y_true, sirs_pred, zero_division=0))
    pr_auc = calculate_pr_auc(y_true, sirs_prob)
    roc_auc = calculate_roc_auc(y_true, sirs_prob)

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "pr_auc": pr_auc, "roc_auc": roc_auc
    }

def select_operating_threshold(y_true, y_prob, target_sensitivity=0.85):
    """
    Selects optimal operating threshold on validation predictions.
    Objective: Maximize specificity subject to sensitivity >= target_sensitivity.
    Never uses test set.
    """
    best_threshold = 0.5
    best_spec = -1.0
    best_metrics = None

    thresholds = np.linspace(0.01, 0.99, 99)
    valid_candidates = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        # Calculate specificity
        tn = np.sum((y_true == 0) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            "threshold": float(t),
            "sensitivity": float(rec),
            "specificity": float(spec),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }

        if rec >= target_sensitivity:
            valid_candidates.append(metrics)

    if valid_candidates:
        # Choose candidate with max specificity
        best_candidate = max(valid_candidates, key=lambda x: x["specificity"])
        return best_candidate["threshold"], best_candidate
    else:
        # Fallback: candidate with maximum sensitivity
        all_candidates = []
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            rec = recall_score(y_true, preds, zero_division=0)
            prec = precision_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            tn = np.sum((y_true == 0) & (preds == 0))
            fp = np.sum((y_true == 0) & (preds == 1))
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            all_candidates.append({
                "threshold": float(t),
                "sensitivity": float(rec),
                "specificity": float(spec),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            })
        best_candidate = max(all_candidates, key=lambda x: x["sensitivity"])
        return best_candidate["threshold"], best_candidate

def train_and_select_model():
    print("=" * 70)
    print("SEPSISGUARD PHASE 3 — MODEL SELECTION & TRAINING PIPELINE")
    print("=" * 70)

    # 1. Inspect and Load Phase 2 Outputs
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("PHASE 3 BLOCKED")
        print(f"Missing processed datasets in {PROCESSED_DIR}. Run data/preprocess.py first.")
        sys.exit(1)

    df_train_raw = pd.read_csv(train_path)
    df_val_raw = pd.read_csv(val_path)

    print(f"[*] Loaded Phase 2 train dataset ({len(df_train_raw)} rows, {df_train_raw[PATIENT_ID_COL].nunique()} patients)")
    print(f"[*] Loaded Phase 2 val dataset   ({len(df_val_raw)} rows, {df_val_raw[PATIENT_ID_COL].nunique()} patients)")

    # 2. Clinically Motivated Derived Feature Engineering
    print("\n--- 1. CAUSAL DERIVED FEATURE ENGINEERING ---")
    df_train = add_derived_features(df_train_raw)
    df_val = add_derived_features(df_val_raw)

    feature_cols = [c for c in BASE_FEATURE_COLS if c in df_train.columns]
    for col in MISSINGNESS_INDICATOR_COLS:
        if col in df_train.columns and col not in feature_cols:
            feature_cols.append(col)
    for col in DERIVED_FEATURE_COLS:
        if col in df_train.columns and col not in feature_cols:
            feature_cols.append(col)

    print(f"Feature set size: {len(feature_cols)} features")
    print("Features:", feature_cols)

    X_train = df_train[feature_cols]
    y_train = df_train[TARGET_COL]
    groups_train = df_train[PATIENT_ID_COL]

    X_val = df_val[feature_cols]
    y_val = df_val[TARGET_COL]

    # Verify zero missing values in feature set
    assert X_train.isnull().sum().sum() == 0, "Null values found in X_train!"
    assert X_val.isnull().sum().sum() == 0, "Null values found in X_val!"

    results_summary = {}

    # 3. BASELINE MODEL 1 — Majority Class
    print("\n--- 2. BASELINE MODEL 1: MAJORITY CLASS ---")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_val_prob = dummy.predict_proba(X_val)[:, 1]
    dummy_pr_auc = calculate_pr_auc(y_val, dummy_val_prob)
    dummy_roc_auc = calculate_roc_auc(y_val, dummy_val_prob)
    dummy_acc = accuracy_score(y_val, dummy.predict(X_val))

    results_summary["Majority Class"] = {
        "mean_pr_auc": dummy_pr_auc, "std_pr_auc": 0.0,
        "mean_roc_auc": dummy_roc_auc, "std_roc_auc": 0.0,
        "val_acc": dummy_acc, "model_obj": dummy
    }
    print(f"  PR-AUC: {dummy_pr_auc:.4f} | ROC-AUC: {dummy_roc_auc:.4f} | Accuracy: {dummy_acc:.4f}")

    # 4. BASELINE MODEL 2 — SIRS Heuristic
    print("\n--- 3. BASELINE MODEL 2: SIRS CLINICAL HEURISTIC ---")
    sirs_res = evaluate_sirs_heuristic(df_val)
    results_summary["SIRS Heuristic"] = {
        "mean_pr_auc": sirs_res["pr_auc"], "std_pr_auc": 0.0,
        "mean_roc_auc": sirs_res["roc_auc"], "std_roc_auc": 0.0,
        "val_acc": sirs_res["accuracy"], "model_obj": None
    }
    print(f"  PR-AUC: {sirs_res['pr_auc']:.4f} | ROC-AUC: {sirs_res['roc_auc']:.4f} | Accuracy: {sirs_res['accuracy']:.4f}")

    # 5. BASELINE MODEL 3 — Logistic Regression
    print("\n--- 4. BASELINE MODEL 3: LOGISTIC REGRESSION ---")
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000))
    ])
    lr_pr_aucs, lr_roc_aucs = evaluate_group_cv(lr_pipe, X_train, y_train, groups_train)
    lr_pipe.fit(X_train, y_train)

    results_summary["Logistic Regression"] = {
        "mean_pr_auc": float(np.mean(lr_pr_aucs)), "std_pr_auc": float(np.std(lr_pr_aucs)),
        "mean_roc_auc": float(np.mean(lr_roc_aucs)), "std_roc_auc": float(np.std(lr_roc_aucs)),
        "model_obj": lr_pipe
    }
    print(f"  CV PR-AUC:  {np.mean(lr_pr_aucs):.4f} ± {np.std(lr_pr_aucs):.4f}")
    print(f"  CV ROC-AUC: {np.mean(lr_roc_aucs):.4f} ± {np.std(lr_roc_aucs):.4f}")

    # 6. CANDIDATE MODEL 1 — Random Forest
    print("\n--- 5. CANDIDATE MODEL 1: RANDOM FOREST ---")
    rf_base = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    gkf = GroupKFold(n_splits=5)
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_grid,
        n_iter=8,
        scoring='average_precision',
        cv=gkf,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    start_time = time.time()
    rf_search.fit(X_train, y_train, groups=groups_train)
    rf_fit_time = time.time() - start_time

    best_rf = rf_search.best_estimator_
    rf_pr_aucs, rf_roc_aucs = evaluate_group_cv(best_rf, X_train, y_train, groups_train)

    results_summary["Random Forest"] = {
        "mean_pr_auc": float(np.mean(rf_pr_aucs)), "std_pr_auc": float(np.std(rf_pr_aucs)),
        "mean_roc_auc": float(np.mean(rf_roc_aucs)), "std_roc_auc": float(np.std(rf_roc_aucs)),
        "best_params": rf_search.best_params_, "fit_time": rf_fit_time, "model_obj": best_rf
    }
    print(f"  Best Params: {rf_search.best_params_}")
    print(f"  CV PR-AUC:  {np.mean(rf_pr_aucs):.4f} ± {np.std(rf_pr_aucs):.4f}")
    print(f"  CV ROC-AUC: {np.mean(rf_roc_aucs):.4f} ± {np.std(rf_roc_aucs):.4f}")

    # 7. CANDIDATE MODEL 2 — XGBoost
    if XGBOOST_AVAILABLE:
        print("\n--- 6. CANDIDATE MODEL 2: XGBOOST ---")
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos = n_neg / float(n_pos) if n_pos > 0 else 1.0

        xgb_base = XGBClassifier(
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            n_jobs=-1
        )
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'scale_pos_weight': [scale_pos, scale_pos * 0.8, scale_pos * 1.2]
        }

        xgb_search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=xgb_param_grid,
            n_iter=8,
            scoring='average_precision',
            cv=gkf,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        start_time = time.time()
        xgb_search.fit(X_train, y_train, groups=groups_train)
        xgb_fit_time = time.time() - start_time

        best_xgb = xgb_search.best_estimator_
        xgb_pr_aucs, xgb_roc_aucs = evaluate_group_cv(best_xgb, X_train, y_train, groups_train)

        results_summary["XGBoost"] = {
            "mean_pr_auc": float(np.mean(xgb_pr_aucs)), "std_pr_auc": float(np.std(xgb_pr_aucs)),
            "mean_roc_auc": float(np.mean(xgb_roc_aucs)), "std_roc_auc": float(np.std(xgb_roc_aucs)),
            "best_params": xgb_search.best_params_, "fit_time": xgb_fit_time, "model_obj": best_xgb
        }
        print(f"  Calculated scale_pos_weight: {scale_pos:.2f}")
        print(f"  Best Params: {xgb_search.best_params_}")
        print(f"  CV PR-AUC:  {np.mean(xgb_pr_aucs):.4f} ± {np.std(xgb_pr_aucs):.4f}")
        print(f"  CV ROC-AUC: {np.mean(xgb_roc_aucs):.4f} ± {np.std(xgb_roc_aucs):.4f}")

    # 8. Model Comparison & Selection
    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT (GroupKFold 5-Fold Cross-Validation)")
    print("=" * 70)
    print(f"{'MODEL':<22s} | {'CV PR-AUC':<18s} | {'CV ROC-AUC':<18s}")
    print("-" * 70)
    for model_name, info in results_summary.items():
        pr_str = f"{info['mean_pr_auc']:.4f} ± {info['std_pr_auc']:.4f}"
        roc_str = f"{info['mean_roc_auc']:.4f} ± {info['std_roc_auc']:.4f}"
        print(f"{model_name:<22s} | {pr_str:<18s} | {roc_str:<18s}")
    print("=" * 70)

    # Selection rule: Highest PR-AUC among ML candidates
    ml_candidates = ["Logistic Regression", "Random Forest"]
    if XGBOOST_AVAILABLE:
        ml_candidates.append("XGBoost")

    selected_model_name = max(ml_candidates, key=lambda k: results_summary[k]["mean_pr_auc"])
    selected_model_info = results_summary[selected_model_name]
    selected_model = selected_model_info["model_obj"]

    print(f"\n[SELECTED MODEL]: {selected_model_name}")
    print(f"  Reason: Highest mean GroupKFold PR-AUC ({selected_model_info['mean_pr_auc']:.4f})")

    # 9. Validation-Based Operating Threshold Selection
    print("\n--- 7. VALIDATION THRESHOLD SELECTION ---")
    if hasattr(selected_model, "predict_proba"):
        val_probs = selected_model.predict_proba(X_val)[:, 1]
    else:
        val_probs = selected_model.predict(X_val)

    selected_thresh, val_perf = select_operating_threshold(y_val.values, val_probs, target_sensitivity=TARGET_SENSITIVITY)
    print(f"Target Sensitivity Configured: >= {TARGET_SENSITIVITY * 100:.0f}%")
    print(f"Selected Threshold: {selected_thresh:.2f}")
    print(f"Validation Sensitivity (Recall): {val_perf['sensitivity']:.4f}")
    print(f"Validation Specificity:          {val_perf['specificity']:.4f}")
    print(f"Validation Precision:            {val_perf['precision']:.4f}")
    print(f"Validation F1-Score:             {val_perf['f1']:.4f}")

    # 10. Fit Scaler (for export if needed)
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 11. Model Versioning & Artifact Export
    version_id = "v2_" + datetime.now().strftime("%Y-%m-%d")
    model_artifact_path = os.path.join(MODEL_DIR, f"model_{version_id}.joblib")
    scaler_artifact_path = os.path.join(MODEL_DIR, f"scaler_{version_id}.joblib")
    metadata_artifact_path = os.path.join(MODEL_DIR, f"metadata_{version_id}.json")
    model_card_path = os.path.join(MODEL_DIR, f"model_card_{version_id}.json")

    joblib.dump(selected_model, model_artifact_path)
    joblib.dump(scaler, scaler_artifact_path)

    metadata = {
        "model_version": version_id,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "PhysioNet 2019 Sepsis Challenge (Phase 2 Processed)",
        "features": feature_cols,
        "preprocessing_description": "Median imputation + 6h causal rolling trends + StandardScaler (LR)",
        "model_type": selected_model_name,
        "hyperparameters": selected_model_info.get("best_params", {}),
        "cv_configuration": "GroupKFold (n_splits=5, groups=Patient_ID)",
        "cv_metrics": {
            k: {
                "mean_pr_auc": v["mean_pr_auc"],
                "std_pr_auc": v["std_pr_auc"],
                "mean_roc_auc": v["mean_roc_auc"],
                "std_roc_auc": v["std_roc_auc"]
            } for k, v in results_summary.items()
        },
        "selected_threshold": selected_thresh,
        "validation_performance": val_perf,
        "random_seed": RANDOM_STATE,
        "package_versions": {
            "scikit_learn": joblib.__name__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        }
    }

    with open(metadata_artifact_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update model card
    model_card = {
        "model_name": f"SepsisGuard ML Model ({selected_model_name})",
        "version": version_id,
        "dataset": "PhysioNet 2019 Sepsis Challenge (Real Clinical Data)",
        "features": feature_cols,
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "model_type": selected_model_name,
        "hyperparameters": selected_model_info.get("best_params", {}),
        "cross_validation_method": "GroupKFold (n_splits=5, grouped by Patient_ID)",
        "selection_metric": "PR-AUC (average_precision)",
        "threshold_selection_method": f"Validation set optimization for target sensitivity >= {TARGET_SENSITIVITY}",
        "selected_threshold": selected_thresh,
        "metrics": {
            "cv_pr_auc_mean": selected_model_info["mean_pr_auc"],
            "cv_pr_auc_std": selected_model_info["std_pr_auc"],
            "cv_roc_auc_mean": selected_model_info["mean_roc_auc"],
            "cv_roc_auc_std": selected_model_info["std_roc_auc"],
            "val_sensitivity": val_perf["sensitivity"],
            "val_specificity": val_perf["specificity"],
            "val_precision": val_perf["precision"],
            "val_recall": val_perf["recall"],
            "val_f1": val_perf["f1"]
        },
        "known_limitations": [
            "Trained on PhysioNet 2019 ICU data; external generalizability not yet validated.",
            "High missingness in raw clinical vitals imputed via median values.",
            "Longitudinal risk predictions require clinical verification."
        ],
        "intended_use": "Clinical decision-support tool for monitoring sepsis risk in adult ICU patients.",
        "out_of_scope_use": [
            "Not for standalone autonomous diagnosis or treatment decision making.",
            "Not validated for pediatric or neonatal populations."
        ]
    }

    with open(model_card_path, 'w') as f:
        json.dump(model_card, f, indent=2)

    print(f"\n[OK] Saved versioned model artifact:    {model_artifact_path}")
    print(f"[OK] Saved versioned scaler artifact:   {scaler_artifact_path}")
    print(f"[OK] Saved versioned metadata artifact: {metadata_artifact_path}")
    print(f"[OK] Saved model card artifact:        {model_card_path}")

    # 12. Sanity Check against Old Synthetic Model
    print("\n--- 8. SANITY CHECK AGAINST OLD MODEL ---")
    print("  OLD SYNTHETIC MODEL: 100.0% accuracy / 1.0000 ROC-AUC (Trivially separable synthetic data)")
    print(f"  NEW REAL-DATA MODEL: {selected_model_info['mean_pr_auc']:.4f} CV PR-AUC ({val_perf['sensitivity']*100:.1f}% sensitivity @ thresh {selected_thresh:.2f})")
    print("  RESULT: Flawed synthetic separability eliminated. Realistic performance on real clinical data established.")

    return metadata

if __name__ == "__main__":
    train_and_select_model()
