# Source Code (`src/`)

This directory contains the production-grade, modular Python source code for the **AI-Powered Student Career & Placement Prediction System**.

## Architectural Principles

The source codebase adheres to industry software engineering and machine learning engineering standards:

1. **Separation of Concerns**: Pipeline components are decoupled into discrete responsibilities (ingestion, preprocessing, feature extraction, model fitting, evaluation, and inference).
2. **Scikit-Learn Compatibility**: Transformers and pipeline components will leverage `BaseEstimator` and `TransformerMixin` interfaces to guarantee standard `fit` / `transform` behavior.
3. **Modularity for Future Extensibility**: Designed so that downstream components (such as Explainable AI, API services, and dashboards) can consume trained models and pipelines via clean programmatic interfaces without requiring architectural refactoring.
4. **Strong Typing & Documentation**: All functions, methods, and classes must include Python type annotations (`typing`) and clear docstrings explaining inputs, outputs, and edge cases.

## Planned Modular Structure (Upcoming Phases)

As the project progresses through development phases, `src/` will be organized into focused modules:

```
src/
├── __init__.py           # Package root and version metadata
├── README.md             # Architecture documentation
├── data/                 # Ingestion, validation, and data loading utilities
│   ├── __init__.py
│   ├── loader.py         # Dataset loading and validation logic
│   └── validator.py      # Schema and data integrity verification
├── features/             # Feature engineering and transformation pipelines
│   ├── __init__.py
│   ├── preprocessor.py   # Imputation, encoding, and scaling pipelines
│   └── engineer.py       # Domain-specific feature engineering (skill indexes)
├── models/               # Model training, tuning, and serialization
│   ├── __init__.py
│   ├── train.py          # Training loop and cross-validation execution
│   └── evaluate.py       # Classification metrics and evaluation reports
└── utils/                # General utilities (logging, config, I/O)
    ├── __init__.py
    └── logger.py         # Standardized application logging
```

## Phase 1 Status

In Phase 1 (Project Foundation), this directory is initialized as a valid Python package (`__init__.py`). Implementation of specific modules will commence in subsequent phases following test-driven and modular development workflows.
