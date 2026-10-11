"""
Pydantic Data Schemas for REST API
AI-Powered Student Career & Placement Prediction System
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StudentProfileInput(BaseModel):
    """Input payload representing a student profile."""
    student_id: Optional[str] = Field(None)
    student_name: Optional[str] = Field(None)
    department: str = Field("CSE")
    tenth_percentage: float = Field(..., ge=50.0, le=100.0)
    twelfth_percentage: float = Field(..., ge=50.0, le=100.0)
    cgpa: float = Field(..., ge=5.0, le=10.0)
    history_of_arrears: int = Field(0, ge=0, le=1)
    current_arrears: int = Field(0, ge=0)
    dsa_score: float = Field(..., ge=0.0, le=100.0)
    java_score: float = Field(..., ge=0.0, le=100.0)
    python_score: float = Field(..., ge=0.0, le=100.0)
    sql_score: float = Field(..., ge=0.0, le=100.0)
    web_dev_score: float = Field(..., ge=0.0, le=100.0)
    aptitude_score: float = Field(..., ge=0.0, le=100.0)
    coding_score: float = Field(..., ge=0.0, le=100.0)
    communication_score: float = Field(..., ge=0.0, le=100.0)
    projects_count: int = Field(2, ge=0, le=10)
    internships_count: int = Field(0, ge=0, le=5)
    certifications_count: int = Field(1, ge=0, le=10)
    leetcode_solved: int = Field(50, ge=0)
    github_profile: int = Field(1, ge=0, le=1)
    linkedin_profile: int = Field(1, ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: int
    status: str
    placement_probability: float
    placement_probability_pct: float
    readiness_tier: str


class FactorItem(BaseModel):
    feature: str
    friendly_name: str
    raw_value: Any
    shap_value: float
    impact: str


class ExplanationResponse(BaseModel):
    base_value: float
    top_positive_factors: List[FactorItem]
    top_negative_factors: List[FactorItem]


class AssessmentFullResponse(BaseModel):
    student_id: Optional[str]
    student_name: Optional[str]
    prediction: PredictionResponse
    explanation: ExplanationResponse
    career_readiness: Dict[str, Any]
