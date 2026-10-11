"""
Unit Tests for Recommendation & Skill-Gap Engine (Phase 6)
AI-Powered Student Career & Placement Prediction System
"""

import pytest

from src.recommendations.engine import (
    CareerRecommendationEngine,
    analyze_student_career_readiness,
    ROLE_SDE,
    ROLE_DATA_ANALYST,
)


@pytest.fixture(scope="module")
def strong_profile() -> dict:
    return {
        "department": "CSE",
        "tenth_percentage": 90.0,
        "twelfth_percentage": 88.0,
        "cgpa": 8.7,
        "history_of_arrears": 0,
        "current_arrears": 0,
        "dsa_score": 85.0,
        "java_score": 75.0,
        "python_score": 80.0,
        "sql_score": 80.0,
        "web_dev_score": 78.0,
        "aptitude_score": 82.0,
        "coding_score": 88.0,
        "communication_score": 78.0,
        "projects_count": 3,
        "internships_count": 2,
        "certifications_count": 2,
        "leetcode_solved": 350,
    }


@pytest.fixture(scope="module")
def struggling_profile() -> dict:
    return {
        "department": "ECE",
        "tenth_percentage": 62.0,
        "twelfth_percentage": 59.0,
        "cgpa": 6.2,
        "history_of_arrears": 1,
        "current_arrears": 2,
        "dsa_score": 30.0,
        "java_score": 35.0,
        "python_score": 38.0,
        "sql_score": 40.0,
        "web_dev_score": 25.0,
        "aptitude_score": 45.0,
        "coding_score": 32.0,
        "communication_score": 50.0,
        "projects_count": 0,
        "internships_count": 0,
        "certifications_count": 0,
        "leetcode_solved": 10,
    }


def test_evaluate_skill_gaps(strong_profile, struggling_profile):
    """Verifies that skill gap calculations correctly distinguish strengths vs critical deficits."""
    engine = CareerRecommendationEngine()

    strong_gaps = engine.evaluate_skill_gaps(strong_profile, target_role=ROLE_SDE)
    assert len(strong_gaps["strengths"]) >= 3
    assert len(strong_gaps["critical_deficits"]) == 0

    struggling_gaps = engine.evaluate_skill_gaps(struggling_profile, target_role=ROLE_SDE)
    assert len(struggling_gaps["critical_deficits"]) >= 2
    assert "Data Structures & Algorithms" in struggling_gaps["critical_deficits"]


def test_compute_role_compatibility(strong_profile):
    """Verifies that all 5 roles receive ranked affinity scores."""
    engine = CareerRecommendationEngine()
    roles = engine.compute_role_compatibility(strong_profile)

    assert len(roles) == 5
    assert roles[0]["compatibility_score_pct"] >= roles[-1]["compatibility_score_pct"]
    assert roles[0]["role_name"] == ROLE_SDE  # Strong profile has SDE affinity


def test_check_company_eligibility(strong_profile, struggling_profile):
    """Verifies corporate tier screening rules."""
    engine = CareerRecommendationEngine()

    strong_elig = engine.check_company_eligibility(strong_profile)
    assert strong_elig["tier1_product"]["eligible"] is True
    assert strong_elig["tier2_solutions"]["eligible"] is True
    assert strong_elig["tier3_services"]["eligible"] is True

    struggling_elig = engine.check_company_eligibility(struggling_profile)
    assert struggling_elig["tier1_product"]["eligible"] is False
    assert len(struggling_elig["tier1_product"]["blockers"]) >= 2


def test_generate_personalized_roadmap(struggling_profile):
    """Verifies 4-week roadmap generation addressing student specific gaps."""
    engine = CareerRecommendationEngine()
    roadmap = engine.generate_personalized_roadmap(struggling_profile, target_role=ROLE_SDE)

    assert len(roadmap) == 4
    assert roadmap[0]["week"] == 1
    # Standing arrears should trigger remedial task
    assert any("arrear" in task.lower() for task in roadmap[0]["action_items"])


def test_analyze_student_career_readiness_end_to_end(strong_profile):
    """Verifies unified career readiness pipeline output."""
    res = analyze_student_career_readiness(strong_profile)

    assert "ranked_roles" in res
    assert "selected_target_role" in res
    assert "skill_gap_analysis" in res
    assert "company_eligibility" in res
    assert "preparation_roadmap" in res
