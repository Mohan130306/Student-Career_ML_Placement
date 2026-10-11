# Final Project Presentation & Review Slide Deck Outline

## AI-Powered Student Career & Placement Prediction System
- **Presentation Title**: An Explainable Machine Learning System for Campus Placement Prediction, Skill-Gap Diagnostics, and Tailored Career Preparation
- **Target Audience**: College Review Panel, Placement Committee, Academic Evaluators
- **Target Duration**: 10 – 15 Minutes (10 Slides)

---

### Slide 1: Title & Project Overview
- **Header**: AI-Powered Student Career & Placement Prediction System
- **Sub-header**: An Explainable, Data-Driven Career Readiness and Skill-Gap Recommendation Platform
- **Presenter**: Mohan S.
- **Key Talking Points**:
  - Introduction to the project as an end-to-end Machine Learning system.
  - Highlights that the goal is not merely "predicting a label", but empowering students with explainable insights and personalized remediation roadmaps.

---

### Slide 2: Real-World Problem Statement
- **Key Pain Points**:
  - **Late Intervention**: At-risk students discovered during recruitment drives when it's too late.
  - **Unidimensional Bias**: Sole focus on CGPA while ignoring coding dexterity, DSA, and problem-solving.
  - **Black-Box Confusion**: Students receive binary rejection without knowing *why* or *what* to fix.
  - **Institutional Challenge**: Placement cells lack automated batch diagnostics to organize targeted training.

---

### Slide 3: Proposed Multi-Tier Solution
- **The 8 Core Deliverables**:
  1. Early placement readiness classification
  2. Calibrated probability scoring
  3. Explainable AI via SHAP
  4. Role-specific skill-gap radar diagnostics
  5. 5-role career track compatibility matching
  6. Corporate tier eligibility screening
  7. Personalized 4-week preparation roadmaps
  8. Institutional batch analytics portal

---

### Slide 4: Multi-Factor Dataset & Governance
- **Dataset Structure (24 Features)**:
  - Academics (10th, 12th, CGPA, Arrears)
  - Technical Skills (DSA, Java, Python, SQL, Web Dev)
  - Core Assessments (Aptitude, Coding Score, Communication)
  - Practical Experience (Projects, Internships, Certifications, LeetCode Solved, GitHub, LinkedIn)
- **Ethical Governance**: Quarantined identifiers (`student_id`), holdout demographic attributes for bias auditing (Adverse Impact Ratio = 0.968).

---

### Slide 5: Leak-Free Preprocessing & Feature Engineering
- **5 Strategic Domain Features**:
  - `fe_technical_aggregate`: Composite engineering breadth.
  - `fe_coding_dexterity`: Normalized coding score + competitive LeetCode volume.
  - `fe_profile_richness`: Portfolio proactivity index.
  - `fe_academic_consistency`: Longitudinal trajectory ratio.
  - `fe_arrear_burden`: Non-linear corporate disqualification penalty.
- **Strict Leakage Protocol**: Stratified 80/20 splitting fitted strictly on training data.

---

### Slide 6: Machine Learning Benchmark & Results
- **Comparison of 4 Diverse Algorithms**:
  - Logistic Regression (L2 Regularized Baseline)
  - Random Forest Classifier (Bagging Ensemble)
  - Gradient Boosted Decision Trees (Sequential Ensemble)
  - XGBoost Classifier (Optimized Gradient Boosting)
- **Champion Selected: Logistic Regression**:
  - **Test ROC-AUC: 0.726** | **Test F1-Score: 0.629** | **Test Accuracy: 67.0%**
  - High generalization stability and superior probability calibration on normalized inputs.

---

### Slide 7: Explainable AI (SHAP) Interpretation
- **Global Drivers**: Internships ($|\phi|=0.41$), Projects ($|\phi|=0.40$), and Portfolio Richness ($|\phi|=0.30$) dominate positive impact. Standing Arrears ($|\phi|=0.27$) is the top penalty.
- **Local Student Explanations**: Decomposes individual predictions into green amplifiers (DSA, LeetCode) and red risk factors (Arrears, Aptitude gap) via waterfall visualizations.

---

### Slide 8: Career Recommendation & Action Roadmap
- **Role Alignment**: Matching against empirical benchmarks for SDE, Data Analyst, Backend, QA, and Systems Support.
- **Corporate Screening**: Three-tier checks (Tier 1 Product, Tier 2 Solutions, Tier 3 IT Services).
- **Personalized 4-Week Roadmap**: Step-by-step weekly action items addressing the student's exact deficit areas.

---

### Slide 9: System Demo & Implementation
- **Interactive Streamlit Web Dashboard**: Live demonstration of Student Self-Assessment Portal and Admin Batch Analytics.
- **FastAPI REST Service**: Production endpoints (`/predict`, `/explain`, `/recommendations`, `/assess`).
- **Database Architecture**: SQLite for local lab testing, seamlessly switchable to PostgreSQL via `.env`.
- **Quality Assurance**: 51 automated unit tests passing (100% pass rate).

---

### Slide 10: Conclusion & Future Scope
- **Summary**: Delivered an end-to-end, validated, explainable career prediction and recommendation system.
- **Future Enhancements**: Automated resume NLP extraction, real-time LeetCode API synchronization, and recruiter candidate shortlisting portal.
- **Q&A**: Open for questions from the evaluation committee.
