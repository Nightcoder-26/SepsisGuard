# -*- coding: utf-8 -*-
"""
Input Validation Schemas (Phase 7 / Phase 10)
Pydantic schemas for REST API input validation and JSON sanitization utilities.
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import Optional

def sanitize(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):   return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):   return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

class VitalsInput(BaseModel):
    Heart_Rate: float = Field(..., ge=20.0, le=300.0)
    Oxygen_Level: float = Field(..., ge=50.0, le=100.0)
    Temperature: float = Field(..., ge=30.0, le=45.0)
    Blood_Pressure: float = Field(..., ge=30.0, le=250.0)
    Resp_Rate: float = Field(..., ge=4.0, le=70.0)
    Age: Optional[float] = Field(65.0, ge=0.0, le=120.0)
    Infection_Marker: Optional[float] = Field(0.5, ge=0.0, le=100.0)
    Mean_Arterial_Pressure: Optional[float] = Field(None, ge=20.0, le=200.0)
    Glucose: Optional[float] = Field(120.0, ge=10.0, le=1000.0)
    Creatinine: Optional[float] = Field(1.0, ge=0.1, le=30.0)
    Platelets: Optional[float] = Field(200.0, ge=1.0, le=1500.0)
    generate_synthesis: Optional[bool] = False

    class Config:
        extra = "ignore"
