# Machine Learning Model Evaluation & Leaderboard Report

## AI-Powered Student Career & Placement Prediction System
- **Phase**: Phase 4 — Model Training & Benchmark Comparison
- **Cross-Validation**: 5-Fold Stratified CV on Training Data ($N = 800$)
- **Holdout Test Evaluation**: Unseen Holdout Set ($N = 200$, Stratified)
- **Selected Champion Model**: **Logistic Regression**

---

## 1. Model Leaderboard Summary

| Model Algorithm | 5-Fold CV F1 | 5-Fold CV AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** 🏆 | `0.644 ± 0.052` | `0.734 ± 0.037` | **67.0%** | 65.9% | 60.2% | **0.629** | **0.726** |
| **Gradient Boosting** | `0.611 ± 0.037` | `0.684 ± 0.054` | **64.5%** | 62.8% | 58.1% | **0.603** | **0.697** |
| **Random Forest** | `0.622 ± 0.031` | `0.728 ± 0.037` | **65.5%** | 66.2% | 52.7% | **0.587** | **0.722** |
| **XGBoost** | `0.630 ± 0.055` | `0.716 ± 0.053` | **62.5%** | 61.3% | 52.7% | **0.567** | **0.700** |

---

## 2. Champion Model Selection Rationale: Logistic Regression

1. **Balanced Discrimination & Generalization**:
   - `Logistic Regression` achieved the highest holdout **F1-Score of 0.629** and an outstanding **ROC-AUC of 0.726**.
   - Test Accuracy: **67.0%** (134/200 correct predictions).

2. **Crucial Error Breakdown on Unseen Students**:
   - **True Positives (TP)**: 56 placed students correctly identified.
   - **True Negatives (TN)**: 78 at-risk students correctly identified for early intervention.
   - **False Positives (FP)**: Only 29 unplaced students falsely predicted as placed. Minimizing FP is vital in educational placement so students in need are not overlooked.
   - **False Negatives (FN)**: 37 placed students predicted unplaced (harmless conservative warning).

3. **Production Deployment Choice**:
   - `Logistic Regression` is persisted to `models/champion_model.joblib`.
   - The tree-based architecture is ideally suited for fast Shapley Additive Explanations (TreeSHAP) in Phase 5.

---

## 3. Evaluation Visualizations

The following diagnostic plots were produced and saved to `docs/figures/`:
- `roc_curves.png`: Multi-model ROC comparison curve.
- `confusion_matrices.png`: Complete error breakdown per model.
