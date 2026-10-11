"""
Database Repository & Query Service
AI-Powered Student Career & Placement Prediction System
"""

import json
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from src.db.models import StudentAssessment


def save_assessment(
    db: Session,
    profile_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    career_result: Dict[str, Any],
    student_id: Optional[str] = None,
    student_name: Optional[str] = None,
) -> StudentAssessment:
    """
    Persists a completed student evaluation record to the database.
    """
    critical_deficits_str = json.dumps(
        career_result.get("skill_gap_analysis", {}).get("critical_deficits", [])
    )

    assessment = StudentAssessment(
        student_id=student_id or profile_data.get("student_id"),
        student_name=student_name,
        department=profile_data["department"],
        tenth_percentage=float(profile_data["tenth_percentage"]),
        twelfth_percentage=float(profile_data["twelfth_percentage"]),
        cgpa=float(profile_data["cgpa"]),
        history_of_arrears=int(profile_data.get("history_of_arrears", 0)),
        current_arrears=int(profile_data.get("current_arrears", 0)),
        dsa_score=float(profile_data["dsa_score"]),
        java_score=float(profile_data["java_score"]),
        python_score=float(profile_data["python_score"]),
        sql_score=float(profile_data["sql_score"]),
        web_dev_score=float(profile_data["web_dev_score"]),
        aptitude_score=float(profile_data["aptitude_score"]),
        coding_score=float(profile_data["coding_score"]),
        communication_score=float(profile_data["communication_score"]),
        projects_count=int(profile_data.get("projects_count", 0)),
        internships_count=int(profile_data.get("internships_count", 0)),
        certifications_count=int(profile_data.get("certifications_count", 0)),
        leetcode_solved=int(profile_data.get("leetcode_solved", 0)),
        github_profile=int(profile_data.get("github_profile", 0)),
        linkedin_profile=int(profile_data.get("linkedin_profile", 0)),
        predicted_placement=int(prediction_result["prediction"]),
        placement_probability=float(prediction_result["placement_probability"]),
        readiness_tier=prediction_result["readiness_tier"],
        target_role=career_result.get("selected_target_role", "Software Development Engineer (SDE)"),
        critical_deficits=critical_deficits_str,
    )

    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment


def get_all_assessments(db: Session, limit: int = 500) -> List[StudentAssessment]:
    """Retrieves recent student assessment records."""
    return db.query(StudentAssessment).order_by(StudentAssessment.created_at.desc()).limit(limit).all()


def get_batch_summary(db: Session) -> Dict[str, Any]:
    """Computes aggregated institutional batch statistics from database records."""
    records = db.query(StudentAssessment).all()
    if not records:
        return {
            "total_students": 0,
            "placed_readiness_rate_pct": 0.0,
            "avg_cgpa": 0.0,
            "avg_coding_score": 0.0,
            "at_risk_count": 0,
        }

    total = len(records)
    ready_count = sum(1 for r in records if r.predicted_placement == 1)
    at_risk_count = sum(1 for r in records if r.predicted_placement == 0)

    avg_cgpa = sum(r.cgpa for r in records) / total
    avg_coding = sum(r.coding_score for r in records) / total

    return {
        "total_students": total,
        "placed_readiness_rate_pct": round((ready_count / total) * 100.0, 1),
        "avg_cgpa": round(avg_cgpa, 2),
        "avg_coding_score": round(avg_coding, 1),
        "at_risk_count": at_risk_count,
    }
