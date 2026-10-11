"""
Model Training, Cross-Validation & Metric Evaluation Module
AI-Powered Student Career & Placement Prediction System

Trains, validates, and compares diverse classification algorithms:
- Logistic Regression (Linear L2 regularized baseline)
- Random Forest Classifier (Ensemble bagging)
- Gradient Boosting Classifier (Sequential boosting)
- XGBoost Classifier (Extreme gradient boosting)

Generates 5-fold Stratified CV metrics, holdout test metrics, ROC plots,
and serializes the champion model artifact.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from src.data.generate_dataset import COL_PLACEMENT_STATUS


def get_candidate_models(random_state: int = 42) -> Dict[str, any]:
    """
    Returns the dictionary of initialized candidate ML models.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, any], Dict[str, any]]:
    """
    Trains candidate models, executes 5-fold Stratified CV, and computes holdout test metrics.

    Args:
        X_train: Processed training features.
        y_train: Training labels.
        X_test: Processed test features.
        y_test: Holdout test labels.
        random_state: Random state for cross-validation splits.

    Returns:
        Tuple[pd.DataFrame, Dict, Dict]:
          - Leaderboard DataFrame
          - Dictionary of trained model instances
          - Dictionary of ROC curve coordinate data
    """
    models = get_candidate_models(random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    leaderboard_records = []
    trained_models = {}
    roc_data = {}

    for name, model in models.items():
        # 1. 5-Fold Stratified Cross-Validation on Training Data
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=cv_scoring,
            return_train_score=False,
        )

        cv_acc_mean = cv_results["test_accuracy"].mean()
        cv_f1_mean = cv_results["test_f1"].mean()
        cv_f1_std = cv_results["test_f1"].std()
        cv_auc_mean = cv_results["test_roc_auc"].mean()
        cv_auc_std = cv_results["test_roc_auc"].std()

        # 2. Fit on Entire Training Set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # 3. Evaluate on Holdout Test Set (Unseen)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": test_auc}

        leaderboard_records.append({
            "model": name,
            "cv_accuracy": round(cv_acc_mean, 4),
            "cv_f1_mean": round(cv_f1_mean, 4),
            "cv_f1_std": round(cv_f1_std, 4),
            "cv_auc_mean": round(cv_auc_mean, 4),
            "cv_auc_std": round(cv_auc_std, 4),
            "test_accuracy": round(test_acc, 4),
            "test_precision": round(test_prec, 4),
            "test_recall": round(test_rec, 4),
            "test_f1": round(test_f1, 4),
            "test_roc_auc": round(test_auc, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        })

    leaderboard_df = pd.DataFrame(leaderboard_records).sort_values(
        by=["test_f1", "test_roc_auc"], ascending=False
    ).reset_index(drop=True)

    return leaderboard_df, trained_models, roc_data


def generate_evaluation_plots(
    roc_data: Dict[str, any],
    trained_models: Dict[str, any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
):
    """
    Renders and exports ROC curves and Confusion Matrix plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Overlay ROC Curves
    plt.figure(figsize=(9, 7), dpi=150)
    palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6"]

    for idx, (name, data) in enumerate(roc_data.items()):
        plt.plot(
            data["fpr"],
            data["tpr"],
            color=palette[idx % len(palette)],
            linewidth=2.0,
            label=f"{name} (AUC = {data['auc']:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.7, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Holdout Test Set ROC Curves Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png")
    plt.close()

    # 2. Confusion Matrices Grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    axes_flat = axes.ravel()

    for idx, (name, model) in enumerate(trained_models.items()):
        ax = axes_flat[idx]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Not Placed", "Placed"],
            yticklabels=["Not Placed", "Placed"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.suptitle("Holdout Test Confusion Matrices", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png")
    plt.close()


def generate_markdown_report(leaderboard: pd.DataFrame, champion_name: str, out_file: Path):
    """
    Compiles comparative metrics and champion selection rationale into docs/model_evaluation.md.
    """
    report = f"""# Machine Learning Model Evaluation & Leaderboard Report

## AI-Powered Student Career & Placement Prediction System
- **Phase**: Phase 4 — Model Training & Benchmark Comparison
- **Cross-Validation**: 5-Fold Stratified CV on Training Data ($N = 800$)
- **Holdout Test Evaluation**: Unseen Holdout Set ($N = 200$, Stratified)
- **Selected Champion Model**: **{champion_name}**

---

## 1. Model Leaderboard Summary

| Model Algorithm | 5-Fold CV F1 | 5-Fold CV AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in leaderboard.iterrows():
        is_champ = " 🏆" if row["model"] == champion_name else ""
        report += (
            f"| **{row['model']}**{is_champ} | "
            f"`{row['cv_f1_mean']:.3f} ± {row['cv_f1_std']:.3f}` | "
            f"`{row['cv_auc_mean']:.3f} ± {row['cv_auc_std']:.3f}` | "
            f"**{row['test_accuracy'] * 100:.1f}%** | "
            f"{row['test_precision'] * 100:.1f}% | "
            f"{row['test_recall'] * 100:.1f}% | "
            f"**{row['test_f1']:.3f}** | "
            f"**{row['test_roc_auc']:.3f}** |\n"
        )

    champ_row = leaderboard[leaderboard["model"] == champion_name].iloc[0]

    report += f"""
---

## 2. Champion Model Selection Rationale: {champion_name}

1. **Balanced Discrimination & Generalization**:
   - `{champion_name}` achieved the highest holdout **F1-Score of {champ_row['test_f1']:.3f}** and an outstanding **ROC-AUC of {champ_row['test_roc_auc']:.3f}**.
   - Test Accuracy: **{champ_row['test_accuracy'] * 100:.1f}%** ({champ_row['true_positives'] + champ_row['true_negatives']}/200 correct predictions).

2. **Crucial Error Breakdown on Unseen Students**:
   - **True Positives (TP)**: {champ_row['true_positives']} placed students correctly identified.
   - **True Negatives (TN)**: {champ_row['true_negatives']} at-risk students correctly identified for early intervention.
   - **False Positives (FP)**: Only {champ_row['false_positives']} unplaced students falsely predicted as placed. Minimizing FP is vital in educational placement so students in need are not overlooked.
   - **False Negatives (FN)**: {champ_row['false_negatives']} placed students predicted unplaced (harmless conservative warning).

3. **Production Deployment Choice**:
   - `{champion_name}` is persisted to `models/champion_model.joblib`.
   - The tree-based architecture is ideally suited for fast Shapley Additive Explanations (TreeSHAP) in Phase 5.

---

## 3. Evaluation Visualizations

The following diagnostic plots were produced and saved to `docs/figures/`:
- `roc_curves.png`: Multi-model ROC comparison curve.
- `confusion_matrices.png`: Complete error breakdown per model.
"""

    out_file.write_text(report, encoding="utf-8")


def run_training_pipeline(
    data_dir: Path,
    model_dir: Path,
    report_file: Path,
    figures_dir: Path,
    random_state: int = 42,
) -> Dict[str, any]:
    """
    Orchestrates loading, training, cross-validation, plotting, and model persistence.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")[COL_PLACEMENT_STATUS]
    y_test = pd.read_csv(data_dir / "y_test.csv")[COL_PLACEMENT_STATUS]

    leaderboard, trained_models, roc_data = evaluate_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state,
    )

    champion_name = leaderboard.iloc[0]["model"]
    champion_model = trained_models[champion_name]

    # Serialize Champion Model
    champion_path = model_dir / "champion_model.joblib"
    joblib.dump(champion_model, champion_path)

    # Serialize all models for archival
    for name, model in trained_models.items():
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, model_dir / f"model_{slug}.joblib")

    # Generate Plots
    generate_evaluation_plots(roc_data, trained_models, X_test, y_test, figures_dir)

    # Generate Markdown Report
    generate_markdown_report(leaderboard, champion_name, report_file)

    # Save CSV leaderboard
    leaderboard.to_csv(data_dir.parent / "processed" / "model_leaderboard.csv", index=False)

    return {
        "champion_name": champion_name,
        "champion_path": str(champion_path),
        "leaderboard": leaderboard,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate student placement models.")
    parser.add_argument("--datadir", type=str, default="data/processed")
    parser.add_argument("--modeldir", type=str, default="models")
    parser.add_argument("--report", type=str, default="docs/model_evaluation.md")
    parser.add_argument("--figdir", type=str, default="docs/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_training_pipeline(
        data_dir=Path(args.datadir),
        model_dir=Path(args.modeldir),
        report_file=Path(args.report),
        figures_dir=Path(args.figdir),
        random_state=args.seed,
    )

    print("=" * 65)
    print("MODEL TRAINING & EVALUATION BENCHMARK COMPLETE")
    print("=" * 65)
    print(f"Champion Model Selected : {results['champion_name']}")
    print(f"Serialized Model Path   : {results['champion_path']}\n")
    print("Model Leaderboard:")
    print(results["leaderboard"][["model", "test_accuracy", "test_f1", "test_roc_auc"]].to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
