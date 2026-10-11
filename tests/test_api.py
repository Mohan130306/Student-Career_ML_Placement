"""
Unit Tests for FastAPI REST Endpoints (Phase 9)
AI-Powered Student Career & Placement Prediction System
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def sample_payload() -> dict:
    return {
        "student_id": "TEST_API_01",
        "student_name": "API Tester",
        "department": "CSE",
        "tenth_percentage": 88.0,
        "twelfth_percentage": 85.0,
        "cgpa": 8.5,
        "history_of_arrears": 0,
        "current_arrears": 0,
        "dsa_score": 82.0,
        "java_score": 75.0,
        "python_score": 80.0,
        "sql_score": 84.0,
        "web_dev_score": 78.0,
        "aptitude_score": 80.0,
        "coding_score": 85.0,
        "communication_score": 76.0,
        "projects_count": 3,
        "internships_count": 1,
        "certifications_count": 2,
        "leetcode_solved": 260,
        "github_profile": 1,
        "linkedin_profile": 1,
    }


def test_health_check():
    """Verifies health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint(sample_payload):
    """Verifies /api/v1/predict returns valid prediction payload."""
    response = client.post("/api/v1/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "placement_probability" in data
    assert "readiness_tier" in data


def test_explain_endpoint(sample_payload):
    """Verifies /api/v1/explain returns SHAP factor breakdowns."""
    response = client.post("/api/v1/explain", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "base_value" in data
    assert "top_positive_factors" in data
    assert "top_negative_factors" in data


def test_unified_assess_endpoint(sample_payload):
    """Verifies /api/v1/assess combines all services into unified response."""
    response = client.post("/api/v1/assess?save_to_db=true", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "TEST_API_01"
    assert "prediction" in data
    assert "explanation" in data
    assert "career_readiness" in data
