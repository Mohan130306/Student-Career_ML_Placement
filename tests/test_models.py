"""
Unit Tests for Model Training & Inference Engine (Phase 4)
AI-Powered Student Career & Placement Prediction System
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.models.train import get_candidate_models, evaluate_models
from src.models.predict import PlacementPredictor, predict_student_placement, TIER_HIGH, TIER_AT_RISK


@pytest.fixture(scope="module")
def train_test_data():
    """Loads processed train and test matrices for test verification."""
    data_dir = Path("data/processed")
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")["placement_status"]
    y_test = pd.read_csv(data_dir / "y_test.csv")["placement_status"]
    return X_train, y_train, X_test, y_test


def test_candidate_models_initialization():
    """Verifies that all 4 expected candidate model architectures are present."""
    models = get_candidate_models()
    expected_names = {"Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"}
    assert set(models.keys()) == expected_names


def test_evaluate_models_leaderboard(train_test_data):
    """Verifies leaderboard structure and metrics computation."""
    X_train, y_train, X_test, y_test = train_test_data
    # Fast evaluation on smaller subsample for test speed
    leaderboard, trained_models, roc_data = evaluate_models(
        X_train=X_train.iloc[:100],
        y_train=y_train.iloc[:100],
        X_test=X_test.iloc[:30],
        y_test=y_test.iloc[:30],
        random_state=42,
    )

    assert len(leaderboard) == 4
    for metric in ["test_accuracy", "test_f1", "test_roc_auc", "cv_f1_mean"]:
        assert metric in leaderboard.columns
        assert leaderboard[metric].between(0.0, 1.0).all()


def test_predictor_high_readiness_student():
    """Verifies inference pipeline on a high-performing student."""
    predictor = PlacementPredictor()

    high_profile = {
        "department": "CSE",
        "tenth_percentage": 92.0,
        "twelfth_percentage": 90.0,
        "cgpa": 9.15,
        "history_of_arrears": 0,
        "current_arrears": 0,
        "dsa_score": 88.0,
        "java_score": 82.0,
        "python_score": 85.0,
        "sql_score": 84.0,
        "web_dev_score": 80.0,
        "aptitude_score": 86.0,
        "coding_score": 90.0,
        "communication_score": 82.0,
        "projects_count": 4,
        "internships_count": 2,
        "certifications_count": 3,
        "leetcode_solved": 320,
        "github_profile": 1,
        "linkedin_profile": 1,
    }

    result = predictor.predict_single(high_profile)
    assert result["prediction"] == 1
    assert result["placement_probability"] > 0.65
    assert result["readiness_tier"] == TIER_HIGH


def test_predictor_at_risk_student():
    """Verifies inference pipeline on a struggling, at-risk student."""
    predictor = PlacementPredictor()

    at_risk_profile = {
        "department": "ECE",
        "tenth_percentage": 58.0,
        "twelfth_percentage": 55.0,
        "cgpa": 5.9,
        "history_of_arrears": 1,
        "current_arrears": 3,
        "dsa_score": 25.0,
        "java_score": 30.0,
        "python_score": 32.0,
        "sql_score": 28.0,
        "web_dev_score": 20.0,
        "aptitude_score": 35.0,
        "coding_score": 22.0,
        "communication_score": 40.0,
        "projects_count": 0,
        "internships_count": 0,
        "certifications_count": 0,
        "leetcode_solved": 5,
        "github_profile": 0,
        "linkedin_profile": 0,
    }

    result = predictor.predict_single(at_risk_profile)
    assert result["prediction"] == 0
    assert result["placement_probability"] < 0.40
    assert result["readiness_tier"] == TIER_AT_RISK


def test_predict_batch():
    """Verifies batch inference on multiple student records."""
    predictor = PlacementPredictor()
    data_dir = Path("data/raw")
    df = pd.read_csv(data_dir / "student_placement_benchmark.csv").head(10)

    scored_df = predictor.predict_batch(df)
    assert "predicted_placement" in scored_df.columns
    assert "placement_probability" in scored_df.columns
    assert "readiness_tier" in scored_df.columns
    assert len(scored_df) == 10
