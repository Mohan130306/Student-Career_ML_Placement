# Test Suite (`tests/`)

This directory houses the automated test suite for the **AI-Powered Student Career & Placement Prediction System**.

## Testing Philosophy for Machine Learning

Reliable machine learning systems require rigorous automated testing across multiple layers:

1. **Unit Testing**:
   - Verification of individual helper functions, data transformers, and custom scikit-learn estimators.
   - Deterministic handling of missing values, edge cases (e.g., zero divisions, unseen categorical labels), and data type enforcements.
2. **Data Integrity & Schema Testing**:
   - Verification of expected column schemas, value boundaries (e.g., GPA ranges, assessment percentages, valid non-negative counts).
   - Assertion that target labels conform strictly to binary classification specifications (`0` and `1`).
3. **Pipeline Integration Testing**:
   - Verification that end-to-end preprocessing pipelines correctly transform raw-like test fixtures into model-ready tensors/arrays without shape mismatches or data leakage.
4. **Model Performance & Regression Testing**:
   - Guardrails to ensure newly retrained models do not degrade below established minimum baseline metrics (e.g., accuracy, F1-score, ROC-AUC).

## Running Tests

Tests are executed using `pytest` within the project virtual environment:

```bash
# Run the entire test suite
pytest

# Run tests with verbose output and test names
pytest -v

# Run tests with code coverage report
pytest --cov=src tests/
```

## Planned Test Structure (Upcoming Phases)

```
tests/
├── __init__.py
├── README.md             # Testing guide
├── conftest.py           # Shared pytest fixtures (mock dataframes, dummy pipelines)
├── test_data_loader.py   # Data validation and loading tests
├── test_preprocessor.py  # Pipeline transformation tests
└── test_model.py         # Model training and prediction contract tests
```

## Phase 1 Status

The testing framework is initialized with `pytest` in `requirements.txt`. Formal test cases will be developed alongside data loading and transformation modules in Phase 2.
