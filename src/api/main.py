"""
FastAPI Application Entry Point & Route Definitions
AI-Powered Student Career & Placement Prediction System
"""

from typing import Any, Dict, Optional
from fastapi import FastAPI, Depends, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.api.schemas import (
    StudentProfileInput,
    PredictionResponse,
    ExplanationResponse,
    AssessmentFullResponse,
)
from src.models.predict import PlacementPredictor
from src.explainability.explainer import PlacementExplainer
from src.recommendations.engine import analyze_student_career_readiness
from src.db.connection import get_db, init_db
from src.db.repository import save_assessment, get_batch_summary, get_all_assessments


# Initialize database schema tables on startup
init_db()

# Lazy loaded singletons
_predictor: Optional[PlacementPredictor] = None
_explainer: Optional[PlacementExplainer] = None


def get_predictor() -> PlacementPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PlacementPredictor()
    return _predictor


def get_explainer() -> PlacementExplainer:
    global _explainer
    if _explainer is None:
        _explainer = PlacementExplainer()
    return _explainer


app = FastAPI(
    title="AI-Powered Student Career & Placement Prediction System",
    description="REST API serving ML predictions, SHAP explainability, and career preparation roadmaps.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "service": "Placement Prediction & Career Analytics API"}


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_placement(profile: StudentProfileInput):
    """Predicts placement readiness and probability for an individual student."""
    predictor = get_predictor()
    data_dict = profile.model_dump()
    result = predictor.predict_single(data_dict)
    return result


@app.post("/api/v1/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain_prediction(profile: StudentProfileInput):
    """Generates individual SHAP feature attributions and risk penalties."""
    explainer = get_explainer()
    data_dict = profile.model_dump()
    explanation = explainer.explain_student(data_dict)
    return explanation


@app.post("/api/v1/recommendations", tags=["Recommendations"])
def get_recommendations(profile: StudentProfileInput, target_role: Optional[str] = None):
    """Computes skill gaps, role compatibility, company eligibility, and tailored roadmap."""
    data_dict = profile.model_dump()
    readiness = analyze_student_career_readiness(data_dict, target_role=target_role)
    return readiness


@app.post("/api/v1/assess", response_model=AssessmentFullResponse, tags=["Unified Assessment"])
def full_assessment(
    profile: StudentProfileInput,
    save_to_db: bool = Query(False, description="Persist assessment in database"),
    db: Session = Depends(get_db),
):
    """Unified endpoint returning prediction, explanation, and career guidance in a single payload."""
    data_dict = profile.model_dump()
    predictor = get_predictor()
    explainer = get_explainer()

    pred_res = predictor.predict_single(data_dict)
    explain_res = explainer.explain_student(data_dict)
    career_res = analyze_student_career_readiness(data_dict)

    if save_to_db:
        save_assessment(
            db=db,
            profile_data=data_dict,
            prediction_result=pred_res,
            career_result=career_res,
            student_id=profile.student_id,
            student_name=profile.student_name,
        )

    return {
        "student_id": profile.student_id,
        "student_name": profile.student_name,
        "prediction": pred_res,
        "explanation": explain_res,
        "career_readiness": career_res,
    }


@app.get("/api/v1/admin/batch-analytics", tags=["Admin Analytics"])
def admin_batch_summary(db: Session = Depends(get_db)):
    """Retrieves batch-level institutional metrics and readiness rates."""
    return get_batch_summary(db)
