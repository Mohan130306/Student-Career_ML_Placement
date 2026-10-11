"""
Database persistence and repository package.
AI-Powered Student Career & Placement Prediction System
"""

from src.db.connection import get_db, init_db, engine, Base
from src.db.models import StudentAssessment
from src.db.repository import save_assessment, get_all_assessments, get_batch_summary

__all__ = [
    "get_db",
    "init_db",
    "engine",
    "Base",
    "StudentAssessment",
    "save_assessment",
    "get_all_assessments",
    "get_batch_summary",
]
