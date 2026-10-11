"""
Explainable AI (XAI) Engine Using SHAP
AI-Powered Student Career & Placement Prediction System

Provides transparent, interpretable machine learning diagnostics:
- Global feature attribution: Overall drivers across the student population
- Local instance explanation: Exact positive amplifiers and risk penalties for any individual student
- Human-interpretable diagnostic statements
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data.generate_dataset import (
    COL_STUDENT_ID,
    COL_GENDER,
    COL_PLACEMENT_STATUS,
    COL_PLACED_ROLE,
)
from src.features.build_features import add_domain_features
from src.features.pipeline import get_feature_names_from_preprocessor


FEATURE_FRIENDLY_NAMES = {
    "tenth_percentage": "10th Grade Percentage",
    "twelfth_percentage": "12th Grade Percentage",
    "cgpa": "University CGPA",
    "history_of_arrears": "History of Arrears",
    "current_arrears": "Active Current Arrears",
    "dsa_score": "DSA Proficiency",
    "java_score": "Java Programming",
    "python_score": "Python Programming",
    "sql_score": "SQL Database Proficiency",
    "web_dev_score": "Web Development",
    "aptitude_score": "General Aptitude Test",
    "coding_score": "Platform Coding Assessment",
    "communication_score": "HR & Communication Rating",
    "projects_count": "Verified Projects Count",
    "internships_count": "Industry Internships Count",
    "certifications_count": "Professional Certifications Count",
    "leetcode_solved": "LeetCode Problems Solved",
    "github_profile": "Active GitHub Presence",
    "linkedin_profile": "Active LinkedIn Presence",
    "fe_technical_aggregate": "Composite Technical Breadth",
    "fe_coding_dexterity": "Coding Dexterity Index",
    "fe_profile_richness": "Portfolio Richness Score",
    "fe_academic_consistency": "Academic Consistency",
    "fe_arrear_burden": "Arrear Burden Penalty",
    "department_AIDS": "Dept: AI & Data Science",
    "department_CSBS": "Dept: CS & Business Systems",
    "department_CSE": "Dept: Computer Science",
    "department_ECE": "Dept: Electronics & Comm.",
    "department_IT": "Dept: Information Technology",
}


class PlacementExplainer:
    """
    Computes SHAP-based global and individual local explanations for student placement predictions.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/champion_model.joblib",
        preprocessor_path: Union[str, Path] = "models/preprocessor.joblib",
        background_data_path: Union[str, Path] = "data/processed/X_train.csv",
        max_background_samples: int = 150,
    ):
        self.model_path = Path(model_path).resolve()
        self.preprocessor_path = Path(preprocessor_path).resolve()
        self.background_data_path = Path(background_data_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
        if not self.background_data_path.exists():
            raise FileNotFoundError(f"Background training data not found at {self.background_data_path}")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.feature_names = get_feature_names_from_preprocessor(self.preprocessor)

        # Load background data for SHAP reference
        X_train_full = pd.read_csv(self.background_data_path)
        if len(X_train_full) > max_background_samples:
            self.X_background = X_train_full.sample(n=max_background_samples, random_state=42)
        else:
            self.X_background = X_train_full

        # Initialize SHAP explainer
        # For linear models (Logistic Regression) LinearExplainer is optimal
        try:
            self.explainer = shap.LinearExplainer(
                self.model,
                masker=shap.maskers.Independent(data=self.X_background),
            )
        except Exception:
            # Fallback for tree-based or general models
            self.explainer = shap.Explainer(self.model, self.X_background)

    def _transform_student(self, student_profile: Dict[str, Any]) -> pd.DataFrame:
        """Transforms a raw student dict into preprocessed feature DataFrame."""
        df = pd.DataFrame([student_profile])
        cols_to_drop = [c for c in [COL_STUDENT_ID, COL_GENDER, COL_PLACEMENT_STATUS, COL_PLACED_ROLE] if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop)
        df_featured = add_domain_features(df_clean)
        X_trans = self.preprocessor.transform(df_featured)
        return pd.DataFrame(X_trans, columns=self.feature_names)

    def explain_global(self, output_fig_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Computes mean absolute SHAP values across the background dataset.

        Returns:
            pd.DataFrame: Global feature importance rankings.
        """
        shap_values = self.explainer(self.X_background)

        # Extract values matrix
        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if vals.ndim == 3:  # (samples, features, classes)
                vals = vals[:, :, 1]
        else:
            vals = np.array(shap_values)

        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        df_importance = pd.DataFrame({
            "feature": self.feature_names,
            "friendly_name": [FEATURE_FRIENDLY_NAMES.get(f, f) for f in self.feature_names],
            "importance": mean_abs_shap,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        if output_fig_path:
            output_fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 8), dpi=150)
            top15 = df_importance.head(15).iloc[::-1]
            plt.barh(top15["friendly_name"], top15["importance"], color="#2563eb", edgecolor="black", linewidth=0.5)
            plt.title("SHAP Global Feature Importance (Top Placement Drivers)", fontsize=13, fontweight="bold", pad=15)
            plt.xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)", fontsize=11)
            plt.tight_layout()
            plt.savefig(output_fig_path)
            plt.close()

        return df_importance

    def explain_student(self, student_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates exact individual SHAP feature contributions for a single student profile.

        Args:
            student_profile: Dictionary containing raw student attributes.

        Returns:
            Dict: Detailed attribution with positive strengths and negative risk factors.
        """
        X_student = self._transform_student(student_profile)
        shap_res = self.explainer(X_student)

        if hasattr(shap_res, "values"):
            values = shap_res.values[0]
            if values.ndim == 2:
                values = values[:, 1]
            base_value = float(shap_res.base_values[0]) if hasattr(shap_res, "base_values") else 0.0
        else:
            values = np.array(shap_res)[0]
            base_value = 0.0

        attributions = []
        for name, shap_val in zip(self.feature_names, values):
            raw_val = student_profile.get(name, "Engineered/Encoded")
            friendly = FEATURE_FRIENDLY_NAMES.get(name, name)
            attributions.append({
                "feature": name,
                "friendly_name": friendly,
                "raw_value": raw_val,
                "shap_value": round(float(shap_val), 4),
                "impact": "Positive (Amplifier)" if shap_val > 0 else "Negative (Risk Penalty)",
            })

        df_attr = pd.DataFrame(attributions)
        positive_factors = df_attr[df_attr["shap_value"] > 0].sort_values(
            by="shap_value", ascending=False
        ).head(5).to_dict("records")

        negative_factors = df_attr[df_attr["shap_value"] < 0].sort_values(
            by="shap_value", ascending=True
        ).head(5).to_dict("records")

        return {
            "base_value": round(base_value, 4),
            "top_positive_factors": positive_factors,
            "top_negative_factors": negative_factors,
            "all_attributions": df_attr.sort_values(by="shap_value", ascending=False).to_dict("records"),
        }

    def plot_student_waterfall(
        self,
        student_profile: Dict[str, Any],
        output_path: Path,
        max_display: int = 10,
    ):
        """
        Renders an individual student SHAP waterfall bar plot.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        explanation = self.explain_student(student_profile)

        # Combine top pos and neg
        items = explanation["top_positive_factors"][:5] + explanation["top_negative_factors"][:5]
        items_sorted = sorted(items, key=lambda x: abs(x["shap_value"]))

        names = [it["friendly_name"] for it in items_sorted]
        values = [it["shap_value"] for it in items_sorted]
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]

        plt.figure(figsize=(9, 6), dpi=150)
        plt.barh(names, values, color=colors, edgecolor="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
        plt.title("Individual Prediction SHAP Feature Attribution", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("SHAP Value (Contribution toward Placement Outcome)", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run SHAP Explainability Engine.")
    parser.add_argument("--figpath", type=str, default="docs/figures/shap_global_importance.png")
    args = parser.parse_args()

    explainer = PlacementExplainer()
    print("Computing global SHAP feature importance...")
    imp_df = explainer.explain_global(output_fig_path=Path(args.figpath))

    print("=" * 65)
    print("SHAP EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 65)
    print("Top 10 Global Predictive Features:")
    for idx, row in imp_df.head(10).iterrows():
        print(f"  {idx+1:02d}. {row['friendly_name']:<35} (SHAP Importance: {row['importance']:.4f})")
    print(f"\nPlot Saved: {args.figpath}")
    print("=" * 65)


if __name__ == "__main__":
    main()
