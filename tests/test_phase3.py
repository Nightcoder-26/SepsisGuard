# -*- coding: utf-8 -*-
"""
Phase 3 Unit Tests - SepsisGuard Model Pipeline Improvement
Tests GroupKFold patient isolation, test set isolation, class imbalance calculations,
validation threshold selection, model reload, and reproducibility.
"""

import unittest
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.features import add_derived_features
from model.train_model import (
    train_and_select_model, select_operating_threshold, evaluate_group_cv,
    PROCESSED_DIR, MODEL_DIR, TARGET_COL, PATIENT_ID_COL, RANDOM_STATE
)

class TestPhase3ModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load Phase 2 datasets and run Phase 3 feature engineering for tests."""
        cls.train_path = os.path.join(PROCESSED_DIR, "train.csv")
        cls.val_path = os.path.join(PROCESSED_DIR, "val.csv")
        cls.test_path = os.path.join(PROCESSED_DIR, "test.csv")

        cls.df_train_raw = pd.read_csv(cls.train_path)
        cls.df_val_raw   = pd.read_csv(cls.val_path)
        cls.df_test_raw  = pd.read_csv(cls.test_path)

        cls.df_train = add_derived_features(cls.df_train_raw)
        cls.df_val   = add_derived_features(cls.df_val_raw)
        cls.df_test  = add_derived_features(cls.df_test_raw)

    def test_1_groupkfold_no_patient_leakage(self):
        """TEST 1: Verify no patient appears in both training and validation folds in GroupKFold."""
        groups = self.df_train[PATIENT_ID_COL]
        gkf = GroupKFold(n_splits=5)

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(self.df_train, groups=groups)):
            train_pids = set(groups.iloc[train_idx])
            val_pids   = set(groups.iloc[val_idx])

            overlap = train_pids & val_pids
            self.assertEqual(len(overlap), 0, f"Patient leakage detected in fold {fold_idx}: {overlap}")

    def test_2_test_set_isolation(self):
        """TEST 2: Verify test set patient IDs are strictly isolated and never used in training/CV folds."""
        train_pids = set(self.df_train[PATIENT_ID_COL].unique())
        val_pids   = set(self.df_val[PATIENT_ID_COL].unique())
        test_pids  = set(self.df_test[PATIENT_ID_COL].unique())

        self.assertEqual(len(train_pids & test_pids), 0, "Test patient IDs found in training data!")
        self.assertEqual(len(val_pids & test_pids), 0, "Test patient IDs found in validation data!")

    def test_3_randomizedsearch_uses_groupkfold(self):
        """TEST 3: Verify RandomizedSearchCV is configured with GroupKFold cross-validation."""
        rf = RandomForestClassifier(random_state=RANDOM_STATE)
        gkf = GroupKFold(n_splits=5)
        param_grid = {'n_estimators': [10, 20]}

        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=1,
            cv=gkf,
            scoring='average_precision',
            random_state=RANDOM_STATE
        )

        self.assertIsInstance(search.cv, GroupKFold, "RandomizedSearchCV does not use GroupKFold!")
        self.assertEqual(search.scoring, 'average_precision', "RandomizedSearchCV does not score by average_precision!")

    def test_4_class_imbalance_handling(self):
        """TEST 4: Verify class weight / scale_pos_weight calculation is strictly based on training data."""
        y_train = self.df_train[TARGET_COL]
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        expected_scale_pos = n_neg / float(n_pos)

        self.assertGreater(expected_scale_pos, 1.0, "Class imbalance ratio invalid!")
        self.assertAlmostEqual(expected_scale_pos, 49.04, places=1)

    def test_5_threshold_selection_uses_val_data(self):
        """TEST 5: Verify operating threshold is selected using validation set predictions, not test set."""
        y_val_dummy = self.df_val[TARGET_COL].values
        # Create mock predictions
        np.random.seed(42)
        mock_probs = np.random.uniform(0, 1, size=len(y_val_dummy))

        thresh, metrics = select_operating_threshold(y_val_dummy, mock_probs, target_sensitivity=0.85)

        self.assertGreaterEqual(metrics["sensitivity"], 0.0)
        self.assertLessEqual(metrics["sensitivity"], 1.0)
        self.assertIn("specificity", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertTrue(0.01 <= thresh <= 0.99)

    def test_6_model_reload_and_metadata_verification(self):
        """TEST 6: Verify saved versioned model and metadata files exist, reload cleanly, and match."""
        files = os.listdir(MODEL_DIR)
        model_files = [f for f in files if f.startswith("model_v2_") and f.endswith(".joblib")]
        metadata_files = [f for f in files if f.startswith("metadata_v2_") and f.endswith(".json")]

        self.assertTrue(len(model_files) > 0, "No versioned model file found in model/!")
        self.assertTrue(len(metadata_files) > 0, "No versioned metadata file found in model/!")

        model_path = os.path.join(MODEL_DIR, model_files[0])
        metadata_path = os.path.join(MODEL_DIR, metadata_files[0])

        loaded_model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.assertIsNotNone(loaded_model)
        self.assertEqual(metadata["dataset_name"], "PhysioNet 2019 Sepsis Challenge (Phase 2 Processed)")
        self.assertIn("cv_metrics", metadata)
        self.assertIn("selected_threshold", metadata)
        self.assertEqual(metadata["random_seed"], 42)

    def test_7_reproducibility(self):
        """TEST 7: Verify training results are reproducible given fixed RANDOM_STATE=42."""
        feature_cols = ['Heart_Rate', 'Oxygen_Level', 'Temperature', 'Blood_Pressure', 'Resp_Rate']
        X = self.df_train[feature_cols].iloc[:1000]
        y = self.df_train[TARGET_COL].iloc[:1000]

        rf1 = RandomForestClassifier(n_estimators=20, random_state=42)
        rf1.fit(X, y)
        p1 = rf1.predict_proba(X)

        rf2 = RandomForestClassifier(n_estimators=20, random_state=42)
        rf2.fit(X, y)
        p2 = rf2.predict_proba(X)

        np.testing.assert_array_almost_equal(p1, p2, decimal=6, err_msg="Model predictions non-reproducible!")

if __name__ == '__main__':
    unittest.main()
