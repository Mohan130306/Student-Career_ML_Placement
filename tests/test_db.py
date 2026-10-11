"""
Unit Tests for Database Persistence & Repository (Phase 7/9)
AI-Powered Student Career & Placement Prediction System
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.connection import Base
from src.db.repository import save_assessment, get_all_assessments, get_batch_summary


@pytest.fixture
def in_memory_db():
    """Provides an isolated in-memory SQLite database session for unit tests."""
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


def test_save_and_retrieve_assessment(in_memory_db):
    """Verifies persisting and querying a student assessment."""
    profile = {
        "student_id": "TEST_001",
        "department": "CSE",
        "tenth_percentage": 88.0,
        "twelfth_percentage": 85.0,
        "cgpa": 8.3,
        "history_of_arrears": 0,
        "current_arrears": 0,
        "dsa_score": 80.0,
        "java_score": 75.0,
        "python_score": 78.0,
        "sql_score": 82.0,
        "web_dev_score": 74.0,
        "aptitude_score": 79.0,
        "coding_score": 81.0,
        "communication_score": 76.0,
        "projects_count": 3,
        "internships_count": 1,
        "certifications_count": 2,
        "leetcode_solved": 220,
    }
    prediction = {
        "prediction": 1,
        "placement_probability": 0.825,
        "readiness_tier": "High Placement Readiness",
    }
    career = {
        "selected_target_role": "Software Development Engineer (SDE)",
        "skill_gap_analysis": {"critical_deficits": ["None"]},
    }

    saved = save_assessment(
        db=in_memory_db,
        profile_data=profile,
        prediction_result=prediction,
        career_result=career,
        student_name="Jane Doe",
    )

    assert saved.id is not None
    assert saved.student_name == "Jane Doe"
    assert saved.predicted_placement == 1
    assert saved.placement_probability == 0.825

    records = get_all_assessments(in_memory_db)
    assert len(records) == 1

    summary = get_batch_summary(in_memory_db)
    assert summary["total_students"] == 1
    assert summary["placed_readiness_rate_pct"] == 100.0
    assert summary["avg_cgpa"] == 8.3
