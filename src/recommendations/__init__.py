"""
Career Recommendation and Skill-Gap Engine Package.
AI-Powered Student Career & Placement Prediction System
"""

from src.recommendations.engine import (
    CareerRecommendationEngine,
    ROLE_BENCHMARKS,
    analyze_student_career_readiness,
)

__all__ = [
    "CareerRecommendationEngine",
    "ROLE_BENCHMARKS",
    "analyze_student_career_readiness",
]
