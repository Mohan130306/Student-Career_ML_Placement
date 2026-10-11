"""
AI-Powered Student Career & Placement Prediction System
Main Interactive Streamlit Application

Provides:
1. Student Self-Assessment Portal (Predictions, SHAP attribution, Skill Gaps, Roadmap)
2. Institutional Placement Cell Analytics Dashboard (Batch KPIs, Department comparisons, At-Risk Roster)
3. Model Diagnostics & Leaderboard Studio
"""

import json
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.models.predict import PlacementPredictor, TIER_HIGH, TIER_MODERATE, TIER_AT_RISK
from src.explainability.explainer import PlacementExplainer
from src.recommendations.engine import (
    CareerRecommendationEngine,
    ROLE_BENCHMARKS,
    ROLE_SDE,
    ROLE_DATA_ANALYST,
    ROLE_BACKEND_FULLSTACK,
    ROLE_QA_TEST,
    ROLE_SYSTEMS_SUPPORT,
)
from src.db.connection import get_db, init_db
from src.db.repository import save_assessment, get_batch_summary, get_all_assessments


# Page Configuration
st.set_page_config(
    page_title="AI Student Career & Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_core_services():
    """Initializes and caches inference, explainability, and recommendation engines."""
    init_db()
    predictor = PlacementPredictor()
    explainer = PlacementExplainer()
    engine = CareerRecommendationEngine()
    return predictor, explainer, engine


try:
    predictor, explainer, rec_engine = load_core_services()
except Exception as e:
    st.error(f"Error loading system ML models: {e}. Please ensure Phase 4 models are trained.")
    st.stop()


# Sidebar Navigation
st.sidebar.title("🎓 Career AI System")
st.sidebar.markdown("**AI-Powered Placement & Readiness Engine**")
menu = st.sidebar.radio(
    "Navigation Portal",
    [
        "Student Self-Assessment",
        "Institutional Batch Analytics",
        "Model Performance & SHAP Studio",
        "Project Documentation",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Runtime: Python 3.11/3.14 | ML: Scikit-learn + SHAP | DB: SQLite/PostgreSQL")


# =====================================================================
# PORTAL 1: STUDENT SELF-ASSESSMENT
# =====================================================================
if menu == "Student Self-Assessment":
    st.markdown('<div class="main-header">Student Career Readiness & Placement Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate placement probability, identify skill deficits, inspect SHAP drivers, and receive a customized 4-week preparation plan.</div>', unsafe_allow_html=True)

    # Preset profiles buttons for instant test
    st.markdown("##### ⚡ Quick Load Benchmark Profiles:")
    preset_cols = st.columns(3)
    preset_chosen = None

    if preset_cols[0].button("🟢 Load High-Readiness Profile (SDE Track)"):
        preset_chosen = {
            "name": "Jane Smith", "id": "STU_2026_0101", "dept": "CSE",
            "cgpa": 8.85, "tenth": 89.0, "twelfth": 87.5, "hist_arr": 0, "curr_arr": 0,
            "dsa": 86.0, "java": 78.0, "python": 82.0, "sql": 80.0, "web": 78.0,
            "apt": 84.0, "coding": 88.0, "comm": 78.0, "proj": 3, "intern": 2, "cert": 3,
            "leetcode": 340, "github": 1, "linkedin": 1, "target_role": ROLE_SDE,
        }

    if preset_cols[1].button("🟡 Load Moderate Profile (Data Track)"):
        preset_chosen = {
            "name": "Rahul Verma", "id": "STU_2026_0204", "dept": "AIDS",
            "cgpa": 7.60, "tenth": 78.0, "twelfth": 74.0, "hist_arr": 0, "curr_arr": 0,
            "dsa": 64.0, "java": 55.0, "python": 78.0, "sql": 76.0, "web": 58.0,
            "apt": 72.0, "coding": 66.0, "comm": 68.0, "proj": 2, "intern": 1, "cert": 1,
            "leetcode": 160, "github": 1, "linkedin": 1, "target_role": ROLE_DATA_ANALYST,
        }

    if preset_cols[2].button("🔴 Load At-Risk Profile (Intervention Needed)"):
        preset_chosen = {
            "name": "Karthik R.", "id": "STU_2026_0312", "dept": "ECE",
            "cgpa": 6.10, "tenth": 62.0, "twelfth": 58.0, "hist_arr": 1, "curr_arr": 2,
            "dsa": 32.0, "java": 35.0, "python": 40.0, "sql": 42.0, "web": 28.0,
            "apt": 48.0, "coding": 34.0, "comm": 52.0, "proj": 0, "intern": 0, "cert": 0,
            "leetcode": 15, "github": 0, "linkedin": 0, "target_role": ROLE_QA_TEST,
        }

    # Input Form
    with st.form("student_profile_form"):
        st.subheader("1. Student Details & Academic History")
        c1, c2, c3, c4 = st.columns(4)
        stu_name = c1.text_input("Student Name", value=preset_chosen["name"] if preset_chosen else "Alex Morgan")
        stu_id = c2.text_input("Registration / Student ID", value=preset_chosen["id"] if preset_chosen else "STU_2026_0045")
        dept = c3.selectbox("Department", ["CSE", "IT", "ECE", "AIDS", "CSBS"], index=0 if not preset_chosen else ["CSE", "IT", "ECE", "AIDS", "CSBS"].index(preset_chosen["dept"]))
        target_role = c4.selectbox(
            "Target Career Track",
            [ROLE_SDE, ROLE_DATA_ANALYST, ROLE_BACKEND_FULLSTACK, ROLE_QA_TEST, ROLE_SYSTEMS_SUPPORT],
            index=0 if not preset_chosen else [ROLE_SDE, ROLE_DATA_ANALYST, ROLE_BACKEND_FULLSTACK, ROLE_QA_TEST, ROLE_SYSTEMS_SUPPORT].index(preset_chosen["target_role"]),
        )

        c5, c6, c7, c8 = st.columns(4)
        cgpa = c5.slider("University CGPA", 5.0, 10.0, float(preset_chosen["cgpa"]) if preset_chosen else 8.1, 0.05)
        tenth_pct = c6.slider("10th Percentage (%)", 50.0, 100.0, float(preset_chosen["tenth"]) if preset_chosen else 84.0, 0.5)
        twelfth_pct = c7.slider("12th Percentage (%)", 50.0, 100.0, float(preset_chosen["twelfth"]) if preset_chosen else 82.0, 0.5)
        curr_arr = c8.number_input("Active Current Arrears", min_value=0, max_value=6, value=preset_chosen["curr_arr"] if preset_chosen else 0)

        hist_arr = 1 if curr_arr > 0 else (c5.selectbox("History of Arrears (Resolved)", ["No (0)", "Yes (1)"], index=preset_chosen["hist_arr"] if preset_chosen else 0) == "Yes (1)")

        st.subheader("2. Technical Domain Ratings (0 – 100)")
        t1, t2, t3, t4, t5 = st.columns(5)
        dsa = t1.slider("DSA", 0.0, 100.0, float(preset_chosen["dsa"]) if preset_chosen else 75.0)
        java = t2.slider("Java", 0.0, 100.0, float(preset_chosen["java"]) if preset_chosen else 70.0)
        python = t3.slider("Python", 0.0, 100.0, float(preset_chosen["python"]) if preset_chosen else 72.0)
        sql = t4.slider("SQL", 0.0, 100.0, float(preset_chosen["sql"]) if preset_chosen else 76.0)
        web = t5.slider("Web Dev", 0.0, 100.0, float(preset_chosen["web"]) if preset_chosen else 70.0)

        st.subheader("3. Core Assessments & Experience Portfolio")
        e1, e2, e3 = st.columns(3)
        aptitude = e1.slider("Aptitude Assessment", 0.0, 100.0, float(preset_chosen["apt"]) if preset_chosen else 75.0)
        coding = e2.slider("Platform Coding Score", 0.0, 100.0, float(preset_chosen["coding"]) if preset_chosen else 78.0)
        communication = e3.slider("HR & Communication", 0.0, 100.0, float(preset_chosen["comm"]) if preset_chosen else 74.0)

        p1, p2, p3, p4, p5, p6 = st.columns(6)
        projects = p1.number_input("Projects Count", 0, 8, value=preset_chosen["proj"] if preset_chosen else 2)
        internships = p2.number_input("Internships", 0, 4, value=preset_chosen["intern"] if preset_chosen else 1)
        certifications = p3.number_input("Certifications", 0, 6, value=preset_chosen["cert"] if preset_chosen else 1)
        leetcode = p4.number_input("LeetCode Solved", 0, 600, value=preset_chosen["leetcode"] if preset_chosen else 180)
        github = 1 if p5.checkbox("Active GitHub", value=bool(preset_chosen["github"] if preset_chosen else True)) else 0
        linkedin = 1 if p6.checkbox("Active LinkedIn", value=bool(preset_chosen["linkedin"] if preset_chosen else True)) else 0

        submitted = st.form_submit_button("🚀 Run Comprehensive Career Analysis")

    if submitted or preset_chosen:
        profile_dict = {
            "student_id": stu_id,
            "department": dept,
            "tenth_percentage": tenth_pct,
            "twelfth_percentage": twelfth_pct,
            "cgpa": cgpa,
            "history_of_arrears": int(hist_arr),
            "current_arrears": curr_arr,
            "dsa_score": dsa,
            "java_score": java,
            "python_score": python,
            "sql_score": sql,
            "web_dev_score": web,
            "aptitude_score": aptitude,
            "coding_score": coding,
            "communication_score": communication,
            "projects_count": projects,
            "internships_count": internships,
            "certifications_count": certifications,
            "leetcode_solved": leetcode,
            "github_profile": github,
            "linkedin_profile": linkedin,
        }

        # Inference & Recommendations
        with st.spinner("Executing model prediction, SHAP attribution, and skill gap calculations..."):
            pred_out = predictor.predict_single(profile_dict)
            explain_out = explainer.explain_student(profile_dict)
            skill_gaps = rec_engine.evaluate_skill_gaps(profile_dict, target_role=target_role)
            role_ranks = rec_engine.compute_role_compatibility(profile_dict)
            eligibility = rec_engine.check_company_eligibility(profile_dict)
            roadmap = rec_engine.generate_personalized_roadmap(profile_dict, target_role=target_role)

        st.markdown("---")

        # Top Executive Output Banner
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Placement Likelihood", f"{pred_out['placement_probability_pct']:.1f}%")
        kpi_col2.metric("Predicted Outcome", pred_out["status"])
        kpi_col3.metric("Readiness Tier", pred_out["readiness_tier"].split(" /")[0])
        kpi_col4.metric("Top Role Match", role_ranks[0]["role_name"].split(" (")[0])

        # Status Alert Badge
        if pred_out["prediction"] == 1:
            st.success(f"✅ **{stu_name}** exhibits strong placement signals with a confidence score of **{pred_out['placement_probability_pct']:.1f}%**.")
        else:
            st.error(f"⚠️ **{stu_name}** is currently flagged as **At-Risk** ({pred_out['placement_probability_pct']:.1f}% readiness). Early remedial intervention is strongly advised.")

        # Interactive Visual Tabs
        tab_skills, tab_shap, tab_elig, tab_roadmap, tab_roles = st.tabs([
            "🎯 Skill-Gap Radar",
            "🔍 Explainable AI (SHAP)",
            "🏢 Company Eligibility",
            "📅 4-Week Action Roadmap",
            "💼 Career Track Fit",
        ])

        with tab_skills:
            st.subheader(f"Competency Radar: Student vs. {target_role} Benchmark")
            
            # Prepare radar chart
            radar_skills = [s["skill_name"] for s in skill_gaps["skill_breakdown"] if s["skill_key"] != "leetcode_solved"]
            stu_scores = [s["student_score"] for s in skill_gaps["skill_breakdown"] if s["skill_key"] != "leetcode_solved"]
            bm_scores = [s["benchmark_score"] for s in skill_gaps["skill_breakdown"] if s["skill_key"] != "leetcode_solved"]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=stu_scores + [stu_scores[0]],
                theta=radar_skills + [radar_skills[0]],
                fill='toself',
                name=f'{stu_name} (Current)',
                line_color='#2563eb',
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=bm_scores + [bm_scores[0]],
                theta=radar_skills + [radar_skills[0]],
                fill='toself',
                name=f'{target_role} Benchmark',
                line_color='#10b981',
                opacity=0.6,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=450,
                margin=dict(l=40, r=40, t=30, b=30),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Table of Gaps
            st.markdown("##### Detailed Competency Breakdown")
            df_gaps = pd.DataFrame(skill_gaps["skill_breakdown"])[
                ["skill_name", "student_score", "benchmark_score", "gap", "status"]
            ]
            st.dataframe(df_gaps, use_container_width=True)

        with tab_shap:
            st.subheader("Explainable AI: Why did the model make this prediction?")
            st.caption("SHAP feature attributions show the exact factors increasing (+) or decreasing (-) placement probability.")

            shap_col1, shap_col2 = st.columns(2)
            with shap_col1:
                st.markdown("##### 🟢 Top Positive Drivers (Strengths)")
                for item in explain_out["top_positive_factors"]:
                    st.markdown(f"- **{item['friendly_name']}**: `+{item['shap_value']:.3f}` log-odds contribution")

            with shap_col2:
                st.markdown("##### 🔴 Top Risk Penalties (Areas Holding Back)")
                for item in explain_out["top_negative_factors"]:
                    st.markdown(f"- **{item['friendly_name']}**: `{item['shap_value']:.3f}` log-odds penalty")

        with tab_elig:
            st.subheader("Corporate Campus Eligibility Screening")
            for key, tier in eligibility.items():
                with st.expander(f"{tier['tier_name']} — {'✅ ELIGIBLE' if tier['eligible'] else '❌ NOT ELIGIBLE'}", expanded=True):
                    if tier["eligible"]:
                        st.success("Candidate satisfies all minimum academic and technical cutoffs for this company tier.")
                    else:
                        st.error("Identified Disqualification Blockers:")
                        for b in tier["blockers"]:
                            st.write(f"• {b}")

        with tab_roadmap:
            st.subheader(f"Personalized 4-Week Actionable Preparation Roadmap ({target_role})")
            for w in roadmap:
                with st.expander(f"Week {w['week']}: {w['theme']}", expanded=(w['week'] == 1)):
                    st.markdown("**Actionable Tasks:**")
                    for task in w["action_items"]:
                        st.markdown(f"- {task}")
                    st.info(f"🎯 **Target Milestone:** {w['expected_outcome']}")

        with tab_roles:
            st.subheader("Career Job-Role Compatibility Ranking")
            df_roles = pd.DataFrame(role_ranks)
            fig_roles = px.bar(
                df_roles,
                x="compatibility_score_pct",
                y="role_name",
                orientation="h",
                color="compatibility_score_pct",
                color_continuous_scale="Blues",
                labels={"compatibility_score_pct": "Compatibility Score (%)", "role_name": "Job Role"},
            )
            fig_roles.update_layout(yaxis=dict(autorange="reversed"), height=350)
            st.plotly_chart(fig_roles, use_container_width=True)

        # Database Save Button
        st.markdown("---")
        if st.button("💾 Save Assessment to College Database"):
            with get_db() as db:
                career_dict = {
                    "selected_target_role": target_role,
                    "skill_gap_analysis": skill_gaps,
                }
                save_assessment(
                    db=db,
                    profile_data=profile_dict,
                    prediction_result=pred_out,
                    career_result=career_dict,
                    student_id=stu_id,
                    student_name=stu_name,
                )
            st.success(f"Assessment record for {stu_name} ({stu_id}) successfully persisted to database!")


# =====================================================================
# PORTAL 2: INSTITUTIONAL BATCH ANALYTICS
# =====================================================================
elif menu == "Institutional Batch Analytics":
    st.markdown('<div class="main-header">Placement Cell & Admin Batch Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive oversight across graduating cohorts, departmental readiness benchmarks, and at-risk student monitoring.</div>', unsafe_allow_html=True)

    # Ingest benchmark batch data
    batch_csv = Path("data/raw/student_placement_benchmark.csv")
    if batch_csv.exists():
        df_batch = pd.read_csv(batch_csv)
    else:
        st.warning("Benchmark dataset not found. Generating default cohort...")
        st.stop()

    # Top KPI Metrics
    total_batch = len(df_batch)
    placed_batch = int(df_batch["placement_status"].sum())
    rate_batch = (placed_batch / total_batch) * 100.0
    avg_batch_cgpa = df_batch["cgpa"].mean()
    avg_batch_coding = df_batch["coding_score"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Batch Size", f"{total_batch:,} Students")
    m2.metric("Placement Readiness Rate", f"{rate_batch:.1f}%")
    m3.metric("Average Batch CGPA", f"{avg_batch_cgpa:.2f}")
    m4.metric("Average Coding Score", f"{avg_batch_coding:.1f} / 100")

    st.markdown("---")

    # Department Comparisons
    c_dept1, c_dept2 = st.columns(2)
    with c_dept1:
        st.subheader("Department Placement Rates")
        dept_stats = df_batch.groupby("department")["placement_status"].mean().reset_index()
        dept_stats["placement_rate_pct"] = dept_stats["placement_status"] * 100.0
        fig_dept = px.bar(
            dept_stats,
            x="department",
            y="placement_rate_pct",
            color="department",
            labels={"placement_rate_pct": "Readiness Rate (%)", "department": "Department"},
            text_auto=".1f",
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    with c_dept2:
        st.subheader("Department Average Coding Scores")
        dept_coding = df_batch.groupby("department")["coding_score"].mean().reset_index()
        fig_coding = px.bar(
            dept_coding,
            x="department",
            y="coding_score",
            color="department",
            labels={"coding_score": "Avg Coding Score", "department": "Department"},
            text_auto=".1f",
        )
        st.plotly_chart(fig_coding, use_container_width=True)

    st.subheader("🚨 Students Requiring Early Placement Intervention")
    at_risk_df = df_batch[
        (df_batch["placement_status"] == 0) | (df_batch["current_arrears"] > 0)
    ][["student_id", "department", "cgpa", "current_arrears", "coding_score", "dsa_score", "aptitude_score"]]

    st.dataframe(at_risk_df.head(50), use_container_width=True)
    st.caption(f"Displaying top 50 of {len(at_risk_df)} at-risk students flagged for targeted placement training.")


# =====================================================================
# PORTAL 3: MODEL PERFORMANCE & SHAP STUDIO
# =====================================================================
elif menu == "Model Performance & SHAP Studio":
    st.markdown('<div class="main-header">Machine Learning Model Performance & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect cross-validation results, ROC curves, confusion matrices, and global feature importance.</div>', unsafe_allow_html=True)

    leaderboard_csv = Path("data/processed/model_leaderboard.csv")
    if leaderboard_csv.exists():
        st.subheader("🏆 Model Leaderboard (5-Fold Cross-Validation vs. Holdout Test)")
        df_lead = pd.read_csv(leaderboard_csv)
        st.dataframe(df_lead, use_container_width=True)

    c_plot1, c_plot2 = st.columns(2)
    roc_img = Path("docs/figures/roc_curves.png")
    cm_img = Path("docs/figures/confusion_matrices.png")

    if roc_img.exists():
        c_plot1.image(str(roc_img), caption="Holdout Test Set ROC Curves Comparison")
    if cm_img.exists():
        c_plot2.image(str(cm_img), caption="Model Confusion Matrices (True vs. Predicted)")

    st.markdown("---")
    st.subheader("Global Feature Attribution (Tree & Linear SHAP)")
    shap_img = Path("docs/figures/shap_global_importance.png")
    if shap_img.exists():
        st.image(str(shap_img), caption="Global Mean |SHAP| Importance Across Features")


# =====================================================================
# PORTAL 4: DOCUMENTATION & METHODOLOGY
# =====================================================================
elif menu == "Project Documentation":
    st.markdown('<div class="main-header">Project Overview & Scientific Methodology</div>', unsafe_allow_html=True)
    doc_path = Path("docs/project-overview.md")
    if doc_path.exists():
        st.markdown(doc_path.read_text(encoding="utf-8"))
    else:
        st.info("Documentation available in `docs/`.")
