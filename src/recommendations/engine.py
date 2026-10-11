"""
Career Recommendation, Skill-Gap Diagnostic & Preparation Roadmap Engine
AI-Powered Student Career & Placement Prediction System

Transforms machine learning predictions into structured, actionable career guidance:
1. Skill-Gap Diagnostic (Quantified deficits against empirical role benchmarks)
2. Career Role Compatibility Matching (Multi-role affinity scoring)
3. Corporate Eligibility Screening (Tier 1 Product, Tier 2 Solutions, Mass IT Services)
4. Personalized 4-to-8 Week Actionable Preparation Roadmap
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.generate_dataset import (
    COL_CGPA,
    COL_10TH_PERCENT,
    COL_12TH_PERCENT,
    COL_CURRENT_ARREARS,
    COL_HISTORY_OF_ARREARS,
    COL_DSA_SCORE,
    COL_JAVA_SCORE,
    COL_PYTHON_SCORE,
    COL_SQL_SCORE,
    COL_WEB_DEV_SCORE,
    COL_APTITUDE_SCORE,
    COL_CODING_SCORE,
    COL_COMMUNICATION_SCORE,
    COL_PROJECTS_COUNT,
    COL_INTERNSHIPS_COUNT,
    COL_CERTIFICATIONS_COUNT,
    COL_LEETCODE_SOLVED,
    ROLE_SDE,
    ROLE_DATA_ANALYST,
    ROLE_BACKEND_FULLSTACK,
    ROLE_QA_TEST,
    ROLE_SYSTEMS_SUPPORT,
)

# Empirically observed role competency benchmarks (from Phase 3 EDA)
ROLE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    ROLE_SDE: {
        "dsa_score": 82.5,
        "coding_score": 83.0,
        "leetcode_solved": 320.0,
        "aptitude_score": 74.0,
        "communication_score": 69.0,
        "sql_score": 68.0,
        "web_dev_score": 65.0,
    },
    ROLE_DATA_ANALYST: {
        "python_score": 75.0,
        "sql_score": 75.0,
        "aptitude_score": 75.0,
        "communication_score": 69.0,
        "coding_score": 66.0,
        "dsa_score": 62.0,
        "leetcode_solved": 170.0,
    },
    ROLE_BACKEND_FULLSTACK: {
        "web_dev_score": 75.0,
        "java_score": 72.0,
        "sql_score": 74.0,
        "coding_score": 70.0,
        "dsa_score": 68.0,
        "aptitude_score": 70.0,
        "communication_score": 68.0,
        "leetcode_solved": 210.0,
    },
    ROLE_QA_TEST: {
        "communication_score": 70.0,
        "aptitude_score": 67.0,
        "sql_score": 60.0,
        "python_score": 58.0,
        "coding_score": 55.0,
        "dsa_score": 56.0,
        "leetcode_solved": 140.0,
    },
    ROLE_SYSTEMS_SUPPORT: {
        "communication_score": 70.0,
        "aptitude_score": 64.0,
        "sql_score": 58.0,
        "python_score": 52.0,
        "coding_score": 51.0,
        "dsa_score": 51.0,
        "leetcode_solved": 110.0,
    },
}

SKILL_DISPLAY_NAMES = {
    "dsa_score": "Data Structures & Algorithms",
    "coding_score": "Problem Solving & Coding",
    "leetcode_solved": "Competitive Problem Volume (LeetCode)",
    "python_score": "Python Programming",
    "sql_score": "SQL & Relational Databases",
    "java_score": "Core / Enterprise Java",
    "web_dev_score": "Web & Full-Stack Development",
    "aptitude_score": "Quantitative & Logical Aptitude",
    "communication_score": "Professional Communication & HR",
}


class CareerRecommendationEngine:
    """
    Evaluates skill gaps, ranks career roles, checks company eligibility,
    and generates personalized preparation roadmaps.
    """

    def __init__(self, benchmarks: Optional[Dict[str, Dict[str, float]]] = None):
        self.benchmarks = benchmarks or ROLE_BENCHMARKS

    def evaluate_skill_gaps(
        self,
        student_profile: Dict[str, Any],
        target_role: str = ROLE_SDE,
    ) -> Dict[str, Any]:
        """
        Computes granular skill gaps against a specific target career role benchmark.

        Args:
            student_profile: Student input dictionary.
            target_role: Target career role to compare against.

        Returns:
            Dict: Quantified skill gaps and categorized deficits/strengths.
        """
        if target_role not in self.benchmarks:
            target_role = ROLE_SDE

        role_bm = self.benchmarks[target_role]
        gaps_list = []
        critical_deficits = []
        moderate_deficits = []
        strengths = []

        for skill, bm_val in role_bm.items():
            stu_val = float(student_profile.get(skill, 0.0))
            gap = bm_val - stu_val
            pct_fulfillment = min(100.0, max(0.0, (stu_val / bm_val) * 100.0)) if bm_val > 0 else 100.0
            friendly = SKILL_DISPLAY_NAMES.get(skill, skill)

            if gap > 15.0:
                status = "Critical Deficit"
                critical_deficits.append(friendly)
            elif gap > 3.0:
                status = "Moderate Deficit"
                moderate_deficits.append(friendly)
            else:
                status = "Proficient / Strength"
                strengths.append(friendly)

            gaps_list.append({
                "skill_key": skill,
                "skill_name": friendly,
                "student_score": round(stu_val, 1),
                "benchmark_score": round(bm_val, 1),
                "gap": round(gap, 1),
                "fulfillment_pct": round(pct_fulfillment, 1),
                "status": status,
            })

        # Sort by largest deficit first
        gaps_list = sorted(gaps_list, key=lambda x: x["gap"], reverse=True)

        return {
            "target_role": target_role,
            "skill_breakdown": gaps_list,
            "critical_deficits": critical_deficits,
            "moderate_deficits": moderate_deficits,
            "strengths": strengths,
        }

    def compute_role_compatibility(self, student_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculates compatibility fit scores across all 5 career job roles.

        Returns:
            List[Dict]: Roles ranked by compatibility score (0 - 100%).
        """
        compatibilities = []

        for role_name, bm in self.benchmarks.items():
            scores = []
            for skill, bm_val in bm.items():
                stu_val = float(student_profile.get(skill, 0.0))
                # Normalized ratio clamped to 1.15
                ratio = min(1.15, stu_val / bm_val) if bm_val > 0 else 1.0
                scores.append(ratio)

            mean_ratio = np.mean(scores)
            fit_pct = min(99.0, max(15.0, mean_ratio * 90.0))

            compatibilities.append({
                "role_name": role_name,
                "compatibility_score_pct": round(fit_pct, 1),
                "fit_tier": "High Alignment" if fit_pct >= 75.0 else ("Moderate Alignment" if fit_pct >= 55.0 else "Low Alignment"),
            })

        return sorted(compatibilities, key=lambda x: x["compatibility_score_pct"], reverse=True)

    def check_company_eligibility(self, student_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Screens student against industry company eligibility criteria across 3 tiers.

        Returns:
            Dict: Pass/Fail status and identified blockers per tier.
        """
        cgpa = float(student_profile.get(COL_CGPA, 0.0))
        tenth = float(student_profile.get(COL_10TH_PERCENT, 0.0))
        twelfth = float(student_profile.get(COL_12TH_PERCENT, 0.0))
        curr_arrears = int(student_profile.get(COL_CURRENT_ARREARS, 0))
        hist_arrears = int(student_profile.get(COL_HISTORY_OF_ARREARS, 0))
        dsa = float(student_profile.get(COL_DSA_SCORE, 0.0))

        # 1. Tier 1: Product Companies (Google, Microsoft, Amazon, Top Startups)
        tier1_blockers = []
        if cgpa < 7.5:
            tier1_blockers.append(f"CGPA {cgpa:.2f} is below minimum cutoff of 7.50")
        if curr_arrears > 0:
            tier1_blockers.append(f"Standing arrears ({curr_arrears}) violates 0-arrear policy")
        if hist_arrears > 1:
            tier1_blockers.append(f"History of arrears ({hist_arrears}) exceeds maximum allowed (1)")
        if tenth < 70.0 or twelfth < 70.0:
            tier1_blockers.append("10th or 12th percentage below 70.0% cutoff")
        if dsa < 65.0:
            tier1_blockers.append(f"DSA proficiency {dsa:.1f} is below competitive screening threshold (65.0)")

        # 2. Tier 2: Tech Solutions & Core IT (Zoho, Thoughtworks, Persistent, Hexaware)
        tier2_blockers = []
        if cgpa < 6.5:
            tier2_blockers.append(f"CGPA {cgpa:.2f} is below minimum cutoff of 6.50")
        if curr_arrears > 1:
            tier2_blockers.append(f"Standing arrears ({curr_arrears}) exceeds maximum of 1")
        if tenth < 60.0 or twelfth < 60.0:
            tier2_blockers.append("10th or 12th percentage below 60.0% cutoff")

        # 3. Tier 3: Mass Recruiters / IT Services (TCS, Infosys, Wipro, Cognizant, Accenture)
        tier3_blockers = []
        if cgpa < 6.0:
            tier3_blockers.append(f"CGPA {cgpa:.2f} is below minimum cutoff of 6.00")
        if curr_arrears > 0:
            tier3_blockers.append(f"Must clear active arrears ({curr_arrears}) before joining date")
        if tenth < 60.0 or twelfth < 60.0:
            tier3_blockers.append("10th or 12th percentage below 60.0% cutoff")

        return {
            "tier1_product": {
                "tier_name": "Tier 1: Product & High-Growth Tech (8+ LPA)",
                "eligible": len(tier1_blockers) == 0,
                "blockers": tier1_blockers,
            },
            "tier2_solutions": {
                "tier_name": "Tier 2: Tech Solutions & Mid-Tier (5 - 8 LPA)",
                "eligible": len(tier2_blockers) == 0,
                "blockers": tier2_blockers,
            },
            "tier3_services": {
                "tier_name": "Tier 3: IT Services & Mass Recruiters (3.5 - 5 LPA)",
                "eligible": len(tier3_blockers) == 0,
                "blockers": tier3_blockers,
            },
        }

    def generate_personalized_roadmap(
        self,
        student_profile: Dict[str, Any],
        target_role: str = ROLE_SDE,
    ) -> List[Dict[str, Any]]:
        """
        Generates a 4-week personalized tactical preparation roadmap based on top gaps.

        Returns:
            List[Dict]: Week-by-week actionable preparation curriculum.
        """
        gap_eval = self.evaluate_skill_gaps(student_profile, target_role)
        curr_arrears = int(student_profile.get(COL_CURRENT_ARREARS, 0))
        dsa = float(student_profile.get(COL_DSA_SCORE, 0.0))
        sql = float(student_profile.get(COL_SQL_SCORE, 0.0))
        apt = float(student_profile.get(COL_APTITUDE_SCORE, 0.0))
        comm = float(student_profile.get(COL_COMMUNICATION_SCORE, 0.0))
        projects = int(student_profile.get(COL_PROJECTS_COUNT, 0))

        roadmap = []

        # Week 1: Foundational Recovery & High-Priority Fill
        w1_tasks = []
        if curr_arrears > 0:
            w1_tasks.append("🚨 Priority 1: Schedule remedial sessions to clear standing academic arrears.")
        if apt < 65.0:
            w1_tasks.append("Aptitude Drills: Practice Quantitative Speed Math (Percentages, Ratios, Speed-Distance-Time).")
        if dsa < 65.0:
            w1_tasks.append("DSA Fundamentals: Master Arrays, Two Pointers, and String manipulation (Solve 15 Easy LeetCode).")
        else:
            w1_tasks.append("DSA Review: Recursion & Binary Search variations (Solve 10 Medium LeetCode).")

        roadmap.append({
            "week": 1,
            "theme": "Core Foundations & Eligibility Stabilization",
            "action_items": w1_tasks,
            "expected_outcome": "Solidified quantitative speed and foundational array problem solving.",
        })

        # Week 2: Intermediate Data Structures & Database Fluency
        w2_tasks = []
        if sql < 65.0:
            w2_tasks.append("SQL Mastery: Write complex INNER/LEFT JOINs, Subqueries, and GROUP BY aggregations.")
        else:
            w2_tasks.append("Advanced SQL: Window functions (ROW_NUMBER, DENSE_RANK) and Indexing optimizations.")
        w2_tasks.append("DSA Phase 2: Linked Lists, Stacks, and Queues with live whiteboard problem solving.")
        if projects < 2:
            w2_tasks.append("Portfolio Sprint: Begin a full-stack CRUD application with GitHub version control.")

        roadmap.append({
            "week": 2,
            "theme": "Data Structures & Database Implementation",
            "action_items": w2_tasks,
            "expected_outcome": "Complete SQL proficiency and linear data structure fluency.",
        })

        # Week 3: Non-Linear Structures & Algorithmic Depth
        w3_tasks = []
        w3_tasks.append("DSA Phase 3: Trees (Binary Search Tree traversals, BFS, DFS) and Heaps.")
        if target_role == ROLE_DATA_ANALYST:
            w3_tasks.append("Data Analytics: Build exploratory data dashboards in Python using Pandas & Seaborn.")
        elif target_role in [ROLE_SDE, ROLE_BACKEND_FULLSTACK]:
            w3_tasks.append("Backend Architecture: REST API design and relational schema normalization.")
        if comm < 68.0:
            w3_tasks.append("Communication Drills: Participate in 2 Mock Group Discussions on technology trends.")

        roadmap.append({
            "week": 3,
            "theme": "Advanced Problem Solving & System Architecture",
            "action_items": w3_tasks,
            "expected_outcome": "Tree traversal mastery and role-specific technical portfolio development.",
        })

        # Week 4: Interview Simulation & Capstone Polish
        w4_tasks = []
        w4_tasks.append("Mock Coding Rounds: Complete 3 timed mock coding assessments on HackerRank/LeetCode.")
        w4_tasks.append("HR / Technical Simulation: Practice the STAR method (Situation, Task, Action, Result) for project viva.")
        w4_tasks.append("Resume & Profile Polish: Update LinkedIn, pin top GitHub repository with clear README.")

        roadmap.append({
            "week": 4,
            "theme": "Campus Placement Drive Simulation & Profile Polish",
            "action_items": w4_tasks,
            "expected_outcome": "Interview confidence, timed test readiness, and recruiters-ready portfolio.",
        })

        return roadmap


def analyze_student_career_readiness(
    student_profile: Dict[str, Any],
    target_role: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive pipeline combining role compatibility, skill-gaps,
    company eligibility, and tailored roadmap.
    """
    engine = CareerRecommendationEngine()
    roles = engine.compute_role_compatibility(student_profile)

    best_role = target_role or roles[0]["role_name"]
    skill_gaps = engine.evaluate_skill_gaps(student_profile, target_role=best_role)
    eligibility = engine.check_company_eligibility(student_profile)
    roadmap = engine.generate_personalized_roadmap(student_profile, target_role=best_role)

    return {
        "ranked_roles": roles,
        "selected_target_role": best_role,
        "skill_gap_analysis": skill_gaps,
        "company_eligibility": eligibility,
        "preparation_roadmap": roadmap,
    }
