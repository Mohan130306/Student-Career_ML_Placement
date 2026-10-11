# Comprehensive Final Project Report

## AI-Powered Student Career & Placement Prediction System
- **Academic Project Submission**: Final Year / Capstone Engineering Project
- **Domain**: Machine Learning, Educational Data Mining, Explainable AI, Full-Stack Engineering

---

## Executive Summary
This project presents an end-to-end, explainable machine learning platform engineered to address the critical challenges of campus placement readiness assessment and student career guidance in higher engineering education. Traditional campus placement evaluation models depend almost exclusively on academic indicators like Grade Point Average (GPA), neglecting practical problem-solving dexterity, software development breadth, competitive programming practice, and communication skills. Furthermore, existing machine learning implementations often act as black-box predictors, offering binary classifications without explaining the underlying reasoning or guiding at-risk students toward remediation.

The developed system integrates ten comprehensive development phases:
1. **Multi-Factor Dataset Design**: 24 features encompassing academics, 5 core technical skill ratings, 3 standardized assessment dimensions, and verified practical portfolios.
2. **Leak-Free Preprocessing**: Scikit-learn ColumnTransformer with 5 domain-engineered composite features.
3. **Exploratory Data Analysis (EDA)**: Empirical discovery of top placement drivers, demographic fairness audits, and ground-truth role competency benchmarks.
4. **Machine Learning Benchmarking**: Cross-validated comparative evaluation of Logistic Regression, Random Forest, Gradient Boosted Decision Trees, and XGBoost (Champion: Logistic Regression with Test ROC-AUC = 0.726, F1 = 0.629).
5. **Explainable AI (SHAP)**: Game-theoretic feature attribution providing global population drivers and local individual waterfall diagnostics.
6. **Career & Skill Recommendation Engine**: Granular skill-gap diagnostics across 5 industry roles, 3-tier corporate eligibility screening, and dynamic 4-to-8 week preparation roadmaps.
7. **Interactive Web Application**: Streamlit dashboard featuring student self-assessment with interactive radar charts and an institutional placement analytics portal.
8. **Production REST API**: FastAPI backend providing typed endpoints for predictions, explanations, and recommendations.
9. **Database-Agnostic Storage**: SQLAlchemy ORM with SQLite for zero-config lab work, seamlessly portable to PostgreSQL.
10. **Quality Assurance**: 51 automated unit tests across 9 test suites ensuring 100% pass rate and zero regressions.

---

## Chapter 1: Introduction & Problem Statement

### 1.1 Context
Campus placement outcomes are vital metrics of institutional efficacy and academic excellence. For students, campus recruitment represents their primary transition into professional employment. However, conventional placement processes face severe structural deficiencies:
- **Late-Stage Risk Discovery**: Under-prepared students are typically identified late in the placement cycle, leaving inadequate time for remedial preparation.
- **Unidimensional Evaluation**: Relying solely on academic marks neglects multi-dimensional indicators such as coding dexterity, problem-solving aptitude, communication skills, and real-world project experience.
- **Lack of Actionable Feedback**: Conventional evaluation delivers binary pass/fail outcomes without pointing out specific skill deficits or matching student strengths to relevant industry roles.

### 1.2 Proposed Solution & Objectives
The **AI-Powered Student Career & Placement Prediction System** resolves these challenges by providing:
1. Objective placement readiness estimation early in the academic cycle.
2. Calibrated placement probability percentages and readiness tier categorizations.
3. Transparent feature attribution using Explainable AI (SHAP).
4. Granular skill-gap diagnostics against empirical job-role benchmarks.
5. Multi-role career compatibility matching across 5 technical job tracks.
6. Automated corporate campus eligibility screening (Tier 1 Product, Tier 2 Solutions, Tier 3 IT Services).
7. Tailored, step-by-step weekly preparation roadmaps.
8. Institutional batch analytics dashboards for placement cells.

---

## Chapter 2: Literature Review & Related Work
Existing research in educational data mining (EDM) has applied algorithms like Naive Bayes, Decision Trees, and Support Vector Machines to placement prediction. However, analysis of published literature identifies three primary limitations:
- **Target Leakage and Overfitting**: Many prior systems include post-placement attributes (e.g., final company round marks) or fabricate data without realistic noise, leading to unrealistically inflated accuracy claims ($>98\%$).
- **Absence of Practical Competency Signals**: Traditional datasets omit competitive programming practice (LeetCode problem counts) and verified public code portfolios (GitHub repository presence).
- **The Black-Box Dilemma**: Standard predictive tools provide predictions without explaining the causal drivers of the decision.

Our framework addresses all three limitations through strict data governance, leak-free preprocessing, multi-dimensional feature modeling, and game-theoretic SHAP interpretability.

---

## Chapter 3: System Architecture & Database Design

### 3.1 Architectural Overview
The system follows a decoupled, 4-tier modular architecture:
1. **Client & Presentation Tier**: Interactive Streamlit web portals for students and institutional administrators.
2. **API & Service Tier**: FastAPI REST endpoints (`/predict`, `/explain`, `/recommendations`, `/assess`, `/admin/batch-analytics`).
3. **Machine Learning & Core Intelligence Tier**: Preprocessing pipelines (`ColumnTransformer`), trained model classifiers, SHAP explainer, and recommendation engines.
4. **Data Persistence Tier**: SQLAlchemy ORM managing relational storage, supporting local SQLite (`data/placement.db`) and production PostgreSQL via environment variables.

### 3.2 Database Schema
The database stores individual student assessment runs in the `student_assessments` table, capturing academics, skill ratings, model outputs (`predicted_placement`, `placement_probability`, `readiness_tier`), selected target roles, and JSON-serialized critical skill deficits.

---

## Chapter 4: Data Engineering, Governance & Preprocessing

### 4.1 Feature Taxonomy (24 Features)
- **Academic Foundation**: 10th %, 12th %, CGPA, History of Arrears, Current Arrears.
- **Technical Skills (0–100)**: DSA, Core Java, Python, SQL, Web Development.
- **Core Assessments (0–100)**: Aptitude Score, Coding Score, Communication Score.
- **Practical Experience**: Projects Count, Internships Count, Certifications Count, LeetCode Solved Count, GitHub Profile (0/1), LinkedIn Profile (0/1).
- **Target & Metadata**: Primary binary target (`placement_status`) and downstream career alignment ground truth (`placed_role`).

### 4.2 Domain-Engineered Features
Five composite indicators capture strategic placement dynamics:
1. `fe_technical_aggregate`: Composite index of DSA (30%), general coding (25%), SQL (15%), Java/Python (15%), and Web Development (15%).
2. `fe_coding_dexterity`: Combines normalized coding platform score with competitive LeetCode problem volume.
3. `fe_profile_richness`: Proactivity score across projects, internships, certifications, GitHub, and LinkedIn.
4. `fe_academic_consistency`: Ratio of 12th percentage to 10th percentage tracking trajectory stability.
5. `fe_arrear_burden`: Non-linear penalty weighting active current arrears ($3\times$) over historical resolved arrears.

### 4.3 Leak-Free Transformation Pipeline
- Train/test splitting (80% train / 20% test, stratified) is performed strictly **before** fitting transformers.
- Continuous features are imputed via median and normalized via `StandardScaler`.
- Categorical department features are one-hot encoded (`OneHotEncoder`).
- Fitted transformers are serialized to `models/preprocessor.joblib`.

---

## Chapter 5: Machine Learning Benchmarking & Results

### 5.1 Experimental Setup
Four diverse classification models were trained and evaluated:
- **Logistic Regression** (L2 Regularized Baseline)
- **Random Forest Classifier** (Ensemble Bagging)
- **Gradient Boosted Decision Trees** (Sequential Boosting)
- **XGBoost Classifier** (Optimized Extreme Gradient Boosting)

### 5.2 Comparative Results

| Model Algorithm | 5-Fold CV F1 | 5-Fold CV AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** 🏆 | **0.639 ± 0.035** | **0.730 ± 0.031** | **67.0%** | **64.0%** | **61.9%** | **0.629** | **0.726** |
| **Gradient Boosting** | 0.589 ± 0.052 | 0.706 ± 0.036 | 64.5% | 62.1% | 58.7% | 0.603 | 0.697 |
| **Random Forest** | 0.573 ± 0.046 | 0.718 ± 0.029 | 65.5% | 65.4% | 53.3% | 0.587 | 0.722 |
| **XGBoost** | 0.571 ± 0.039 | 0.698 ± 0.034 | 62.5% | 59.8% | 53.8% | 0.567 | 0.700 |

### 5.3 Selection Rationale
Logistic Regression achieved the highest holdout **F1-Score (0.629)** and **ROC-AUC (0.726)**, demonstrating balanced precision and recall on unseen students with excellent probability calibration. It was chosen as the champion model and serialized to `models/champion_model.joblib`.

---

## Chapter 6: Explainable AI via SHAP

### 6.1 Mathematical Formulation
Shapley values allocate credit to features according to their marginal contributions across all feature permutations:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

### 6.2 Empirical Insights
- **Global Population Drivers**: Internships ($|\phi|=0.407$), Projects ($|\phi|=0.399$), and Portfolio Richness ($|\phi|=0.302$) are the primary drivers of placement success. Active Current Arrears ($|\phi|=0.274$) constitutes the dominant negative penalty.
- **Individual Waterfall Attribution**: Each student receives a decomposed explanation showing the exact factors pushing their probability up or down, fostering transparency and trust.

---

## Chapter 7: Skill-Gap Diagnostic & Recommendation Engine
- **Role Benchmarks**: Evaluates student competencies against empirical benchmarks for 5 roles (SDE, Data Analyst, Backend Developer, QA Engineer, Systems Support).
- **Company Eligibility**: Evaluates eligibility across Tier 1 Product (8+ LPA), Tier 2 Solutions (5-8 LPA), and Tier 3 IT Services (3.5-5 LPA) with explicit blocker reporting.
- **Actionable Roadmap**: Constructs a dynamic 4-week preparation curriculum targeting the candidate's highest quantified skill gaps.

---

## Chapter 8: Software Implementation & Web Application
- **Streamlit Application (`app.py`)**: Features a student self-assessment portal with interactive Plotly radar charts, quick benchmark profile loaders, and an institutional batch analytics portal.
- **FastAPI Service (`src/api/main.py`)**: Asynchronous REST API providing typed endpoints for prediction, explanation, and career guidance.
- **Database Layer (`src/db/`)**: Database-agnostic SQLAlchemy ORM defaulting to local SQLite and seamlessly compatible with PostgreSQL.

---

## Chapter 9: Quality Assurance & Testing
The system includes **51 automated unit tests** across 9 test modules:
- Data audit & Excel loading: 18 tests
- Benchmark generator: 6 tests
- Preprocessing & feature pipeline: 4 tests
- EDA engine: 5 tests
- ML training & inference: 5 tests
- Explainable AI (SHAP): 3 tests
- Recommendation engine: 5 tests
- Database repository: 1 test
- FastAPI REST endpoints: 4 tests
- **Overall Status**: 51 passed in 17.8s (100% pass rate).

---

## Chapter 10: Conclusion & Future Work
This project successfully designed and delivered an end-to-end, explainable machine learning system for student career readiness and placement prediction. By uniting leak-free data pipelines, regularized classification, SHAP interpretability, and personalized roadmaps, the framework bridges predictive machine learning with actionable educational remediation.

### Future Enhancements
- Natural language processing (NLP) to parse student resumes and GitHub repositories automatically.
- Live competitive coding platform API synchronization (LeetCode / HackerRank API).
- Recruiter portal for filtering candidate cohorts based on competency benchmarks.
