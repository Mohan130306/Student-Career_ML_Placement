"""
Unit Tests for Explainable AI (SHAP) Module (Phase 5)
AI-Powered Student Career & Placement Prediction System
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.explainability.explainer import PlacementExplainer


@pytest.fixture(scope="module")
def sample_student_profile() -> dict:
    """Fixture providing a realistic student profile."""
    return {
        "department": "CSE",
        "tenth_percentage": 85.0,
        "twelfth_percentage": 82.0,
        "cgpa": 8.4,
        "history_of_arrears": 0,
        "current_arrears": 0,
        "dsa_score": 80.0,
        "java_score": 75.0,
        "python_score": 78.0,
        "sql_score": 82.0,
        "web_dev_score": 76.0,
        "aptitude_score": 78.0,
        "coding_score": 82.0,
        "communication_score": 75.0,
        "projects_count": 3,
        "internships_count": 1,
        "certifications_count": 2,
        "leetcode_solved": 240,
        "github_profile": 1,
        "linkedin_profile": 1,
    }


def test_global_shap_importance():
    """Verifies that global SHAP computation produces valid feature ranking."""
    explainer = PlacementExplainer()
    df_imp = explainer.explain_global()

    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 29
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns
    assert (df_imp["importance"] >= 0.0).all()


def test_student_local_explanation(sample_student_profile: dict):
    """Verifies that individual student SHAP attribution is correctly decomposed."""
    explainer = PlacementExplainer()
    res = explainer.explain_student(sample_student_profile)

    assert "base_value" in res
    assert "top_positive_factors" in res
    assert "top_negative_factors" in res
    assert len(res["top_positive_factors"]) > 0

    first_pos = res["top_positive_factors"][0]
    assert "friendly_name" in first_pos
    assert "shap_value" in first_pos
    assert first_pos["shap_value"] > 0.0


def test_student_waterfall_plot_export(sample_student_profile: dict, tmp_path: Path):
    """Verifies waterfall figure export."""
    explainer = PlacementExplainer()
    out_img = tmp_path / "waterfall.png"
    explainer.plot_student_waterfall(sample_student_profile, output_path=out_img)
    assert out_img.exists()
