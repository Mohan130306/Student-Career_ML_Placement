# Notebooks Directory

This directory contains Jupyter notebooks used for interactive research, exploratory data analysis (EDA), data visualization, and model experimentation for the **AI-Powered Student Career & Placement Prediction System**.

## Purpose

Jupyter notebooks serve as a rapid prototyping and visual diagnostic workspace. They allow data scientists and engineers to inspect distributions, explore feature correlations, and iteratively benchmark initial hypotheses.

## Planned Notebook Workflow

Future phases will introduce structured, numbered notebooks:

1. `01_exploratory_data_analysis.ipynb`: Univariate, bivariate, and multivariate analysis of student academic, technical, and extracurricular profiles.
2. `02_data_preprocessing_pipeline.ipynb`: Missing value imputation, outlier handling, categorical encoding, and feature scaling exploration.
3. `03_feature_engineering_and_selection.ipynb`: Derivation of skill-gap metrics, composite academic-technical indexes, and feature importance filtering.
4. `04_model_training_and_benchmarking.ipynb`: Baseline classification models (Logistic Regression, Random Forest, etc.) and cross-validation comparisons.
5. `05_model_evaluation_and_diagnostics.ipynb`: Confusion matrices, ROC-AUC, Precision-Recall curves, error case analysis.

## Development Rules & Best Practices

- **Code Refactoring**: Logic tested and proven in notebooks must be refactored into modular classes and functions in `src/` to prevent duplicate code and ensure testability.
- **Reproducibility**: All experiments must set explicit random seeds (`random_state=42`).
- **Clean Execution**: Notebooks should run sequentially from top to bottom without relying on hidden cell states. Clear cell outputs or restart kernels to ensure clean execution.
- **Current Status**: In Phase 1, this directory is initialized and ready for interactive experiments once datasets are integrated in Phase 2.
