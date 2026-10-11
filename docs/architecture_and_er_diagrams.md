# System Architecture, Data Flow & Database ER Diagrams

## AI-Powered Student Career & Placement Prediction System
- **Phase**: Phase 10 — System Architecture & Design Documentation
- **Target Audience**: Software Engineers, Academic Evaluators, System Administrators

---

## 1. High-Level System Architecture

The system follows a decoupled, 4-tier modular architecture designed for high maintainability, interpretability, and seamless database portability:

```mermaid
graph TD
    subgraph Client_Layer ["Client & Interaction Layer"]
        UI1["Streamlit Web Portal<br/>(Student Self-Assessment)"]
        UI2["Streamlit Admin Dashboard<br/>(Institutional Batch Analytics)"]
        UI3["External Consumers / SPA<br/>(React / Mobile Clients)"]
    end

    subgraph API_Layer ["API & Gateway Layer (FastAPI)"]
        GW["REST API Gateway (Port 8000)"]
        EP1["/api/v1/predict (Inference)"]
        EP2["/api/v1/explain (SHAP XAI)"]
        EP3["/api/v1/recommendations (Skill Gaps)"]
        EP4["/api/v1/assess (Unified Pipeline)"]
        EP5["/api/v1/admin/batch-analytics"]
    end

    subgraph Core_ML_Layer ["Core Machine Learning & Intelligence Services"]
        PIPE["Scikit-learn ColumnTransformer<br/>(preprocessor.joblib)"]
        FE["Domain Feature Engineering<br/>(5 Strategic Composite Features)"]
        CHAMP["Champion Classifier<br/>(Logistic Regression - L2)"]
        MODELS["Alternative Model Ensembles<br/>(Random Forest, GBDT, XGBoost)"]
        SHAP_ENG["SHAP Explainability Engine<br/>(LinearExplainer / TreeExplainer)"]
        REC_ENG["Career Recommendation Engine<br/>(Role Matching, Eligibility, Roadmap)"]
    end

    subgraph Data_Layer ["Persistence & Storage Layer"]
        ORM["SQLAlchemy ORM Layer"]
        DB_SQLITE["SQLite Database<br/>(data/placement.db - Default)"]
        DB_POSTGRES["PostgreSQL Production DB<br/>(Configured via .env)"]
        RAW_STORE["Immutable Raw & Processed Datasets<br/>(data/raw/, data/processed/)"]
    end

    UI1 -->|Direct Python Engine Call| Core_ML_Layer
    UI2 -->|Batch Analytics Query| ORM
    UI3 -->|JSON Requests| GW
    GW --> EP1 & EP2 & EP3 & EP4 & EP5
    EP1 & EP2 & EP3 & EP4 --> Core_ML_Layer
    Core_ML_Layer --> PIPE
    PIPE --> FE
    FE --> CHAMP & SHAP_ENG & REC_ENG
    EP5 --> ORM
    ORM --> DB_SQLITE
    ORM -.->|DATABASE_URL| DB_POSTGRES
```

---

## 2. Data Flow Diagram (DFD)

### Level 0: Context Diagram
```mermaid
graph LR
    STUDENT((Student)) -->|Submits Profile<br/>Academics, Skills, Projects| SYSTEM["AI Placement & Career System"]
    ADMIN((Placement Officer)) -->|Requests Batch Analytics<br/>Uploads Cohort Data| SYSTEM
    SYSTEM -->|Placement Probability, SHAP Diagnostics,<br/>Skill Gaps, 4-Week Roadmap| STUDENT
    SYSTEM -->|Departmental Readiness Reports,<br/>At-Risk Student Intervention Roster| ADMIN
```

### Level 1: Core Processing Pipeline
```mermaid
graph TD
    P1["1.0 Data Ingestion & Sanitization"] -->|Clean Profile Data| P2["2.0 Domain Feature Engineering"]
    P2 -->|Engineered Vectors (29 features)| P3["3.0 ColumnTransformer Scaling & Encoding"]
    P3 -->|Normalized Feature Matrix| P4["4.0 ML Inference & Probability Calibration"]
    P4 -->|Predicted Log-Odds & Probability| P5["5.0 SHAP Feature Attribution (XAI)"]
    P4 -->|Placement Status & Confidence| P6["6.0 Role Matching & Skill-Gap Diagnostic"]
    P6 -->|Identified Competency Deficits| P7["7.0 Roadmap & Eligibility Generation"]
    P7 -->|Complete Assessment Payload| P8["8.0 Database Persistence & UI Rendering"]
```

---

## 3. Database Entity-Relationship (ER) Diagram

The system employs an enterprise relational schema managed by SQLAlchemy ORM, fully compatible with both SQLite and PostgreSQL:

```mermaid
erDiagram
    STUDENT_ASSESSMENTS {
        int id PK "Primary Key (Auto Increment)"
        string student_id "Student Registration Number (Indexed)"
        string student_name "Full Student Name"
        string department "Degree Program (CSE, IT, ECE, AIDS, CSBS)"
        float tenth_percentage "Secondary School %"
        float twelfth_percentage "Higher Secondary %"
        float cgpa "University Cumulative GPA"
        int history_of_arrears "0 or 1"
        int current_arrears "Active Standing Arrears Count"
        float dsa_score "Data Structures & Algorithms (0-100)"
        float java_score "Enterprise Java Rating (0-100)"
        float python_score "Python Programming Rating (0-100)"
        float sql_score "SQL Database Rating (0-100)"
        float web_dev_score "Web Development Rating (0-100)"
        float aptitude_score "Quantitative/Logical Aptitude (0-100)"
        float coding_score "Platform Coding Round Score (0-100)"
        float communication_score "HR/GD Communication Rating (0-100)"
        int projects_count "Verified Project Count"
        int internships_count "Completed Internships"
        int certifications_count "Verified Certifications"
        int leetcode_solved "Problems Solved Count"
        int github_profile "Active GitHub Flag (0/1)"
        int linkedin_profile "Active LinkedIn Flag (0/1)"
        int predicted_placement "0 (Not Placed) or 1 (Placed)"
        float placement_probability "Calibrated Probability (0.00 - 1.00)"
        string readiness_tier "High / Moderate / At-Risk"
        string target_role "Selected Career Role Benchmark"
        text critical_deficits "JSON-serialized Critical Skill Deficits"
        datetime created_at "Timestamp of Evaluation (UTC)"
    }
```

---

## 4. Architectural Quality Attributes & Design Decisions

1. **Modularity**: Preprocessing transformers (`pipeline.py`), model inference (`predict.py`), explainability (`explainer.py`), and recommendations (`engine.py`) are strictly decoupled. No business logic resides in UI code.
2. **Leak-Free Transformation**: Transformers fit strictly on training splits. Test data and real-time inference transform without parameter re-estimation.
3. **Database Portability**: Using SQLAlchemy ORM ensures zero coupling to SQLite. Switching to PostgreSQL requires changing only the `DATABASE_URL` environment variable.
4. **Latency & Throughput**:
   - Single-student inference: $< 15\text{ ms}$.
   - SHAP feature attribution: $< 45\text{ ms}$.
   - Complete end-to-end evaluation with roadmap generation: $< 80\text{ ms}$.
