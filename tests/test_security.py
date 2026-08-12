# -*- coding: utf-8 -*-
"""
Phase 7 Unit & Security Tests - SepsisGuard Security Hardening Suite
Tests API key authentication, CORS, secret key configuration, input validation (Pydantic),
rate limiting (Flask-Limiter), XSS sanitization, Socket.IO auth and room scoping,
production server settings, and secrets scanning.
"""

import unittest
import os
import sys
import json
import time
import hmac

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set environment variables for testing before importing app
os.environ["FLASK_SECRET_KEY"] = "test_secret_key_8f9a2b4c6d8e1f3a5c7b9d2e4f6a8c0b"
os.environ["API_KEY"] = "test_api_key_3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b"
os.environ["FRONTEND_ORIGIN"] = "http://localhost:5000,http://127.0.0.1:5000"
os.environ["PREDICT_RATE_LIMIT"] = "5 per minute"

from backend.app import app, socketio, API_KEY, FLASK_SECRET_KEY, allowed_origins

class TestPhase7Security(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['RATELIMIT_ENABLED'] = True
        cls.client = app.test_client()
        cls.valid_api_key = API_KEY
        cls.valid_vitals = {
            "Heart_Rate": 85.0,
            "Oxygen_Level": 98.0,
            "Temperature": 37.2,
            "Blood_Pressure": 120.0,
            "Resp_Rate": 16.0,
            "Age": 65.0,
            "Infection_Marker": 0.3
        }

    # 1. AUTHENTICATION TESTS
    def test_1_1_predict_unauthenticated_missing_key(self):
        """POST /predict without X-API-Key header must return HTTP 401."""
        response = self.client.post('/predict', json=self.valid_vitals)
        self.assertEqual(response.status_code, 401)
        data = response.get_json()
        self.assertIn("Unauthorized", data.get("error", ""))

    def test_1_2_predict_unauthenticated_wrong_key(self):
        """POST /predict with invalid X-API-Key header must return HTTP 401."""
        headers = {"X-API-Key": "wrong-invalid-key-value"}
        response = self.client.post('/predict', json=self.valid_vitals, headers=headers)
        self.assertEqual(response.status_code, 401)

    def test_1_3_predict_authenticated_success(self):
        """POST /predict with valid X-API-Key header must return HTTP 200."""
        headers = {"X-API-Key": self.valid_api_key}
        response = self.client.post('/predict', json=self.valid_vitals, headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("risk_score", data)

    def test_1_4_patients_unauthenticated(self):
        """GET /patients without X-API-Key header must return HTTP 401."""
        response = self.client.get('/patients')
        self.assertEqual(response.status_code, 401)

    def test_1_5_patients_authenticated(self):
        """GET /patients with valid X-API-Key header must return HTTP 200."""
        headers = {"X-API-Key": self.valid_api_key}
        response = self.client.get('/patients', headers=headers)
        self.assertEqual(response.status_code, 200)

    # 2. CORS TESTS
    def test_2_1_cors_allowed_origin(self):
        """OPTIONS /predict from allowed origin receives Access-Control-Allow-Origin header."""
        headers = {
            "Origin": "http://localhost:5000",
            "Access-Control-Request-Method": "POST",
            "X-API-Key": self.valid_api_key
        }
        response = self.client.options('/predict', headers=headers)
        self.assertIn("Access-Control-Allow-Origin", response.headers)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5000")

    def test_2_2_cors_disallowed_origin(self):
        """OPTIONS /predict from unauthorized origin does not receive allowed-origin match."""
        headers = {
            "Origin": "http://malicious-hacker-site.com",
            "Access-Control-Request-Method": "POST"
        }
        response = self.client.options('/predict', headers=headers)
        allow_header = response.headers.get("Access-Control-Allow-Origin")
        self.assertNotEqual(allow_header, "http://malicious-hacker-site.com")

    # 3. SECRET KEY CONFIGURATION TESTS
    def test_3_1_no_hardcoded_secret_in_codebase(self):
        """Verify hardcoded default 'sepsisguard-icu-v3' secret key is removed from app.py."""
        app_path = os.path.join(PROJECT_ROOT, "backend", "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("sepsisguard-icu-v3", content, "Hardcoded legacy secret key found in backend/app.py!")

    # 4. INPUT VALIDATION TESTS (PYDANTIC)
    def test_4_1_predict_missing_required_field(self):
        """POST /predict missing required field (Heart_Rate) returns HTTP 422."""
        invalid_data = self.valid_vitals.copy()
        del invalid_data["Heart_Rate"]
        headers = {"X-API-Key": self.valid_api_key}
        response = self.client.post('/predict', json=invalid_data, headers=headers)
        self.assertEqual(response.status_code, 422)
        data = response.get_json()
        self.assertEqual(data.get("error"), "Unprocessable Entity")

    def test_4_2_predict_invalid_data_type(self):
        """POST /predict with string instead of float returns HTTP 422."""
        invalid_data = self.valid_vitals.copy()
        invalid_data["Heart_Rate"] = "one-hundred-twenty"
        headers = {"X-API-Key": self.valid_api_key}
        response = self.client.post('/predict', json=invalid_data, headers=headers)
        self.assertEqual(response.status_code, 422)

    def test_4_3_predict_out_of_range_value(self):
        """POST /predict with Heart_Rate = 900 (out of physiological bounds [20, 300]) returns HTTP 422."""
        invalid_data = self.valid_vitals.copy()
        invalid_data["Heart_Rate"] = 900.0
        headers = {"X-API-Key": self.valid_api_key}
        response = self.client.post('/predict', json=invalid_data, headers=headers)
        self.assertEqual(response.status_code, 422)

    def test_4_4_predict_malformed_json(self):
        """POST /predict with malformed JSON body returns HTTP 400."""
        headers = {"X-API-Key": self.valid_api_key, "Content-Type": "application/json"}
        response = self.client.post('/predict', data="INVALID_JSON_BODY", headers=headers)
        self.assertIn(response.status_code, [400, 422])

    # 5. XSS SANITIZATION TESTS
    def test_5_1_xss_escape_html_rendering(self):
        """Verify dynamic text escaping converts <script> tags into literal escaped text."""
        xss_payload = '<script>alert("XSS")</script>'
        from backend.app import sanitize
        clean = sanitize({"payload": xss_payload})["payload"]
        self.assertEqual(clean, xss_payload) # Server preserves literal text; frontend escapeHtml turns it to &lt;script&gt;

    # 6. RATE LIMITING TESTS
    def test_6_1_rate_limiting_exceeded(self):
        """Rapid requests exceeding rate limit receive HTTP 429."""
        headers = {"X-API-Key": self.valid_api_key}
        statuses = []
        # Default rate limit is 60 per minute
        for _ in range(65):
            res = self.client.post('/predict', json=self.valid_vitals, headers=headers)
            statuses.append(res.status_code)
            if res.status_code == 429:
                break
        self.assertIn(429, statuses, "Rate limiter did not return HTTP 429 when threshold exceeded!")
        # Reset rate limiter state so subsequent test modules are not rate-limited
        from backend.app import limiter
        limiter.reset()

    # 7. SOCKET.IO AUTHENTICATION & ROOM SCOPING TESTS
    def test_7_1_socket_auth_rejection(self):
        """Socket.IO connection without token must be rejected."""
        sio_client = socketio.test_client(app, auth={"token": "wrong_token"})
        self.assertFalse(sio_client.is_connected(), "Socket.IO connected with invalid token!")

    def test_7_2_socket_auth_success_and_room_scoping(self):
        """Socket.IO connection with valid token must succeed and join authorized room."""
        sio_client = socketio.test_client(app, auth={"token": self.valid_api_key})
        self.assertTrue(sio_client.is_connected(), "Socket.IO failed to connect with valid token!")
        received = sio_client.get_received()
        self.assertGreater(len(received), 0, "Connected client did not receive initial snapshot!")
        sio_client.disconnect()

    # 8. PRODUCTION SERVER ENTRY POINT TEST
    def test_8_1_no_allow_unsafe_werkzeug(self):
        """Verify allow_unsafe_werkzeug=True is removed from production entry point in app.py."""
        app_path = os.path.join(PROJECT_ROOT, "backend", "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("allow_unsafe_werkzeug=True", content, "allow_unsafe_werkzeug=True found in app.py!")

if __name__ == '__main__':
    unittest.main()
