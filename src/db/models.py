"""
SQLAlchemy ORM Data Models
AI-Powered Student Career & Placement Prediction System
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, Text
from src.db.connection import Base


def utc_now():
    return datetime.now(timezone.utc)


class StudentAssessment(Base):
    """
    Persisted record of an individual student assessment,
    prediction outcome, and career recommendations.
    """
    __tablename__ = "student_assessments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(String(50), index=True, nullable=True)
    student_name = Column(String(120), nullable=True)
    department = Column(String(20), index=True, nullable=False)

    # Academics
    tenth_percentage = Column(Float, nullable=False)
    twelfth_percentage = Column(Float, nullable=False)
    cgpa = Column(Float, nullable=False)
    history_of_arrears = Column(Integer, default=0)
    current_arrears = Column(Integer, default=0)

    # Technical Competencies
    dsa_score = Column(Float, nullable=False)
    java_score = Column(Float, nullable=False)
    python_score = Column(Float, nullable=False)
    sql_score = Column(Float, nullable=False)
    web_dev_score = Column(Float, nullable=False)

    # Assessments
    aptitude_score = Column(Float, nullable=False)
    coding_score = Column(Float, nullable=False)
    communication_score = Column(Float, nullable=False)

    # Practical
    projects_count = Column(Integer, default=0)
    internships_count = Column(Integer, default=0)
    certifications_count = Column(Integer, default=0)
    leetcode_solved = Column(Integer, default=0)
    github_profile = Column(Integer, default=0)
    linkedin_profile = Column(Integer, default=0)

    # Model Outputs
    predicted_placement = Column(Integer, nullable=False)
    placement_probability = Column(Float, nullable=False)
    readiness_tier = Column(String(80), nullable=False)
    target_role = Column(String(80), nullable=False)
    critical_deficits = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
