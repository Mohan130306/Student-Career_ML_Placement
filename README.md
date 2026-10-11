# AI-Powered Student Career & Placement Prediction System

An enterprise-grade, end-to-end Machine Learning system engineered to evaluate multi-dimensional student profiles—spanning academic records, technical proficiencies, aptitude evaluations, coding performance, communication ratings, projects, internships, and certifications—to predict campus placement outcomes, explain algorithmic drivers via **SHAP**, diagnose **skill deficits**, evaluate **corporate eligibility**, and deliver **personalized career preparation roadmaps**.

---

## 📌 Problem Being Addressed

Campus placement outcomes are critical indicators of institutional effectiveness and student success. However, academic institutions and placement cells often face substantial hurdles:

- **Late-Stage Risk Discovery**: Struggling students are typically identified late in the placement cycle, leaving inadequate time for remedial preparation.
- **Unidimensional Evaluation**: Relying solely on academic marks (e.g., GPA) neglects critical multi-dimensional indicators such as coding dexterity, problem-solving aptitude, communication skills, and real-world project experience.
- **Lack of Actionable Feedback**: Conventional evaluation delivers binary pass/fail outcomes without pointing out specific skill deficits or matching student strengths to relevant industry roles.

---

## 💡 System Capabilities

1. **Placement Readiness Classification**: Unbiased binary prediction of placement outcome (`Placement-Ready` vs. `At-Risk / Needs Intervention`).
2. **Calibrated Probability Index**: Granular probability score ($0.0 - 100.0\%$) and readiness tier categorization.
3. **Explainable AI (SHAP)**: Transparent feature attribution decomposing exact positive amplifiers and negative risk penalties pushing the model log-odds.
4. **Skill-Gap Diagnostic**: Granular gap analysis ($0 - 100\%$) comparing candidate competencies against empirical industry benchmarks for 5 career roles.
5. **Career Track Compatibility Alignment**: Multi-role ranking across **SDE**, **Data Analyst**, **Backend / Full Stack Developer**, **QA / Test Engineer**, and **Systems / Support Engineer**.
6. **Corporate Tier Eligibility Screening**: Automated eligibility checks across **Tier 1 Product (8+ LPA)**, **Tier 2 Tech Solutions (5-8 LPA)**, and **Tier 3 IT Services (3.5-5 LPA)** with explicit disqualification blocker reporting.
7. **Personalized 4-to-8 Week Preparation Roadmap**: Step-by-step weekly action items addressing the student's highest deficit areas.
8. **Institutional Placement Analytics Portal**: Batch-wide monitoring for college administration, departmental readiness comparisons, and at-risk student monitoring rosters.
9. **Dual Architecture (Streamlit + FastAPI)**: Includes both an interactive Streamlit UI dashboard and production-ready REST API endpoints.
10. **Database-Agnostic Storage**: Out-of-the-box local **SQLite** storage (`data/placement.db`) with zero config, effortlessly portable to **PostgreSQL** via `.env`.

---

## 🛠️ Complete Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language Runtime** | **Python 3.11 / 3.14** | Core programming environment |
| **Data Manipulation** | **Pandas**, **NumPy**, **OpenPyXL** | Ingestion, feature transformations, schema validation |
| **Machine Learning** | **Scikit-learn**, **XGBoost** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| **Explainable AI** | **SHAP** | Individual waterfall feature attribution & global importance |
| **Data Visualization** | **Plotly**, **Seaborn**, **Matplotlib** | Radar competency charts, correlation bars, ROC curves |
| **Interactive Dashboard** | **Streamlit** | Multi-portal web application (Student & Admin views) |
| **Backend REST API** | **FastAPI** + **Pydantic** | Production inference and recommendation endpoints |
| **Database & ORM** | **SQLAlchemy** + **SQLite** / **PostgreSQL** | Portable database persistence |
| **Quality Assurance** | **Pytest** (51 passing tests) | Full unit and pipeline regression coverage |

---

## 📂 Project Architecture

```
student-career-placement-ml/
│
├── data/
│   ├── raw/                 # Source benchmarks (student_placement_benchmark.csv & .xlsx)
│   ├── processed/           # Leak-free stratified splits (X_train, X_test, y_train, y_test)
│   └── placement.db         # Local SQLite database (portable to PostgreSQL)
│
├── docs/
│   ├── figures/             # ROC curves, confusion matrices, SHAP plots, radar charts
│   ├── dataset_schema.md    # Complete 24-feature data dictionary and governance
│   ├── data_preprocessing.md# Pipeline architecture and engineered features
│   ├── eda_report.md        # Comprehensive exploratory analysis and role benchmarks
│   ├── model_evaluation.md # Benchmark leaderboard and champion selection report
│   └── explainability_and_recommendations.md # SHAP and recommendation engine specs
│
├── models/
│   ├── preprocessor.joblib  # Serialized ColumnTransformer pipeline
│   ├── champion_model.joblib# Serialized champion model (Logistic Regression)
│   └── model_*.joblib       # Archived Random Forest, Gradient Boosting, XGBoost models
│
├── src/
│   ├── analysis/            # EDA profiling engine (src.analysis.eda)
│   ├── api/                 # Production FastAPI endpoints (src.api.main)
│   ├── data/                # Data loader, audit module, and benchmark generator
│   ├── db/                  # SQLAlchemy models and repository CRUD
│   ├── explainability/      # SHAP explainer module (src.explainability.explainer)
│   ├── features/            # Domain feature engineering & ColumnTransformer pipeline
│   ├── models/              # Training, cross-validation & inference (predict.py)
│   └── recommendations/     # Skill-gap, role matching, eligibility & roadmap engine
│
├── tests/                   # 51 automated unit tests across all 9 subsystems
├── app.py                   # Main interactive Streamlit application
├── requirements.txt         # Pinned production dependencies
└── pytest.ini               # Pytest configuration
```

---

## 🚀 Getting Started

### 1. Run Automated Test Suite
Verify that all 51 unit tests pass:
```bash
pytest
```

### 2. Launch the Interactive Web Dashboard (Streamlit)
To launch the interactive Student Self-Assessment & Admin Analytics dashboard:
```bash
streamlit run app.py
```
*Open your browser at `http://localhost:8501` to use the Student Portal, Radar Charts, SHAP Explanations, and Admin Batch Analytics.*

### 3. Launch the Backend REST API (FastAPI)
To run the high-performance REST API:
```bash
uvicorn src.api.main:app --reload --port 8000
```
*Interactive Swagger documentation will be available at `http://localhost:8000/docs`.*

---

## 💻 Seamless Portability to Personal Laptop (PostgreSQL)

This repository is designed to be **100% portable**:
- **On this machine**: Runs out of the box using local SQLite (`data/placement.db`) with zero setup.
- **On your personal laptop**: To switch to **PostgreSQL**, simply create a `.env` file at the root:
  ```env
  DATABASE_URL=postgresql://postgres:your_password@localhost:5432/placement_db
  ```
  The exact same SQLAlchemy models, API endpoints, and Streamlit app will automatically connect and create tables on your PostgreSQL database!
