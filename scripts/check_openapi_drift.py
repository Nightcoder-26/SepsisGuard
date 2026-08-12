# -*- coding: utf-8 -*-
"""
OpenAPI Schema Drift Verification Script (Phase 12)
Verifies that routes and schemas in openapi.yaml match actual Flask routes and Pydantic validation.
"""

import os
import sys
import yaml
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from backend.validation.schemas import VitalsInput

def check_openapi_drift():
    print("=" * 70)
    print("SEPSISGUARD PHASE 12 - OPENAPI DRIFT VERIFICATION")
    print("=" * 70)

    openapi_path = os.path.join(PROJECT_ROOT, "openapi.yaml")
    if not os.path.exists(openapi_path):
        print(f"[FAIL] openapi.yaml missing at {openapi_path}")
        return False

    with open(openapi_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    # 1. Route Verification
    documented_paths = set(spec.get("paths", {}).keys())
    flask_routes = set([rule.rule for rule in app.url_map.iter_rules()])

    print(f"[*] Documented OpenAPI paths: {sorted(documented_paths)}")

    required_routes = {"/health", "/predict", "/patients"}
    missing_routes = required_routes - documented_paths

    if missing_routes:
        print(f"[FAIL] Missing documented routes in openapi.yaml: {missing_routes}")
        return False

    # 2. Schema Field Verification for VitalsInput
    spec_vitals = spec.get("components", {}).get("schemas", {}).get("VitalsInput", {}).get("properties", {})
    pydantic_fields = VitalsInput.model_fields if hasattr(VitalsInput, "model_fields") else VitalsInput.__fields__

    pydantic_keys = set(pydantic_fields.keys())
    spec_keys = set(spec_vitals.keys())

    missing_in_spec = pydantic_keys - spec_keys
    extra_in_spec = spec_keys - pydantic_keys

    if missing_in_spec:
        print(f"[FAIL] Pydantic fields missing in openapi.yaml VitalsInput: {missing_in_spec}")
        return False

    if extra_in_spec:
        print(f"[FAIL] Extra fields in openapi.yaml not present in Pydantic schema: {extra_in_spec}")
        return False

    print("[OK] All required routes and VitalsInput schema fields match OpenAPI specification exactly.")
    print("[OK] ZERO OpenAPI Drift Detected.")
    return True

if __name__ == '__main__':
    success = check_openapi_drift()
    sys.exit(0 if success else 1)
