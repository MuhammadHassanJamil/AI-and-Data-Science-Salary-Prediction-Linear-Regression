"""
AI & Data Science Salary Prediction System - Interactive Web App
==================================================================
Machine Learning Fundamentals (AIC354) - Assignment 3 (Task 2)
Supervised by: Dr. Rao Muhammad Adeel Nawab
COMSATS University Islamabad, Lahore Campus

This application allows users to predict the annual compensation (MSRP/USD)
of AI & Data Science professionals using a trained regularized Linear
Regression pipeline (Lasso Regression, Rank 1), evaluated using 2/3 train
and 1/3 test partitions with 3-Fold Cross-Validation.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI & DS Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563eb, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #166534;
    }
    .badge-info {
        background-color: #dbeafe;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "salary_prediction_model.pkl"

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_trained_model()

# Category definitions
JOB_TITLES = [
    "Machine Learning Engineer",
    "Deep Learning Engineer",
    "AI Researcher",
    "Data Scientist",
    "Data Engineer",
    "MLOps Engineer",
    "Analytics Manager",
    "Research Scientist",
    "Business Intelligence Analyst",
    "Data Analyst"
]

EXPERIENCE_LEVELS = ["Entry", "Mid", "Senior", "Executive"]
EMPLOYMENT_TYPES = ["Full-time", "Contract", "Freelance", "Part-time"]
COMPANY_SIZES = ["Small", "Medium", "Large"]
LOCATIONS = [
    "United States",
    "United Kingdom",
    "Canada",
    "Germany",
    "Singapore",
    "Australia",
    "Netherlands",
    "France",
    "Spain",
    "India"
]

# ---------------------------------------------------------------------------
# Sidebar / Header
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=70)
    st.markdown("### 💼 AI/DS Salary AI")
    st.markdown("""
    **Model:** Regularized Linear Regression (`Lasso`)  
    **Evaluation:** 2/3 Train, 1/3 Test + 3-Fold CV  
    **Holdout R²:** **0.9303**  
    **Mean RMSE:** **$12,122**
    """)
    st.divider()
    st.markdown("**Academic Context:**")
    st.caption("COMSATS University Islamabad, Lahore Campus")
    st.caption("Course: Machine Learning Fundamentals (AIC354)")
    st.caption("Instructor: Dr. Rao Muhammad Adeel Nawab")
    st.divider()
    st.info("💡 **Tip:** Adjust experience and location to see how geographic market rates impact tech compensation.")

# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">💼 AI & Data Science Salary Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict market-aligned compensation for AI, ML, and Data roles using regularized Linear Regression models.</div>', unsafe_allow_html=True)

# Tabs
tab_calc, tab_benchmark, tab_weights, tab_dataset = st.tabs([
    "🧮 Interactive Salary Predictor",
    "📊 5 Linear Models Benchmark",
    "⚖️ Learned Feature Drivers",
    "📁 Dataset Overview"
])

with tab_calc:
    st.markdown("#### Enter Candidate & Position Specifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        job_title = st.selectbox("Role / Job Title", JOB_TITLES, index=0)
        experience_level = st.selectbox("Experience Level", EXPERIENCE_LEVELS, index=2)
        work_year = st.selectbox("Forecast Work Year", [2023, 2024, 2025, 2026], index=3)
        
    with col2:
        company_location = st.selectbox("Company Location", LOCATIONS, index=0)
        employee_residence = st.selectbox("Employee Residence", LOCATIONS, index=0)
        company_size = st.selectbox("Company Size", COMPANY_SIZES, index=1)
        
    with col3:
        employment_type = st.selectbox("Employment Type", EMPLOYMENT_TYPES, index=0)
        remote_option = st.selectbox("Work Arrangement", ["100% Fully Remote", "50% Hybrid", "0% On-site"], index=0)
        remote_ratio = 100.0 if "100%" in remote_option else (50.0 if "50%" in remote_option else 0.0)
        
        st.write("")
        st.write("")
        predict_btn = st.button("🚀 Calculate Estimated Salary", type="primary", use_container_width=True)

    if predict_btn or True:  # calculate on load and update on click
        input_data = pd.DataFrame([{
            "work_year": work_year,
            "remote_ratio": remote_ratio,
            "experience_level": experience_level,
            "employment_type": employment_type,
            "job_title": job_title,
            "employee_residence": employee_residence,
            "company_location": company_location,
            "company_size": company_size,
        }])
        
        predicted_salary = float(model.predict(input_data)[0])
        rmse_ci = 12122.0  # based on test RMSE
        lower_bound = max(25000, predicted_salary - 1.96 * rmse_ci)
        upper_bound = predicted_salary + 1.96 * rmse_ci
        avg_salary = 146646.41
        diff_pct = ((predicted_salary - avg_salary) / avg_salary) * 100

        st.divider()
        st.markdown("### 🎯 Estimated Market Compensation")
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        
        with mcol1:
            st.metric(
                label="Predicted Annual Salary (USD)",
                value=f"${predicted_salary:,.0f}",
                delta=f"{diff_pct:+.1f}% vs Industry Avg"
            )
            
        with mcol2:
            st.metric(
                label="95% Confidence Lower Bound",
                value=f"${lower_bound:,.0f}"
            )
            
        with mcol3:
            st.metric(
                label="95% Confidence Upper Bound",
                value=f"${upper_bound:,.0f}"
            )
            
        with mcol4:
            tier = "Tier 1: High Compensation" if predicted_salary > 180000 else ("Tier 2: Competitive" if predicted_salary > 120000 else "Tier 3: Emerging")
            st.metric(
                label="Market Band",
                value=tier
            )

        with st.expander("🔍 Inspect Formatted Input Vector Sent to Pipeline"):
            st.dataframe(input_data, use_container_width=True)

with tab_benchmark:
    st.markdown("### 🏆 Comparison of 5 Linear Regression Algorithms")
    st.markdown("""
    All models were evaluated using **2/3 Train, 1/3 Test holdout partitioning** and **3-Fold Cross-Validation** (where each fold trains on 2/3 and tests on 1/3 of the dataset).
    *Notice that none of these regularized/robust formulations were used in the Car Price Prediction system.*
    """)
    
    benchmark_df = pd.DataFrame([
        {"Rank": 1, "Algorithm": "Lasso Regression", "Mathematical Formulation": "L1 Regularization (Feature Sparsity)", "Mean CV RMSE ($)": 12122.58, "Std RMSE ($)": 229.47, "Holdout R²": 0.9303, "Holdout MAE ($)": 9421.16, "Time (s)": 0.16},
        {"Rank": 2, "Algorithm": "Ridge Regression", "Mathematical Formulation": "L2 Regularization (Weight Shrinkage)", "Mean CV RMSE ($)": 12134.99, "Std RMSE ($)": 222.08, "Holdout R²": 0.9300, "Holdout MAE ($)": 9444.18, "Time (s)": 0.07},
        {"Rank": 3, "Algorithm": "Huber Regressor", "Mathematical Formulation": "Robust Regression (Huber Loss)", "Mean CV RMSE ($)": 12135.20, "Std RMSE ($)": 224.84, "Holdout R²": 0.9301, "Holdout MAE ($)": 9435.77, "Time (s)": 1.22},
        {"Rank": 4, "Algorithm": "Bayesian Ridge", "Mathematical Formulation": "Probabilistic Bayesian Priors", "Mean CV RMSE ($)": 12138.48, "Std RMSE ($)": 222.36, "Holdout R²": 0.9300, "Holdout MAE ($)": 9448.61, "Time (s)": 0.11},
        {"Rank": 5, "Algorithm": "ElasticNet Regression", "Mathematical Formulation": "Convex L1 + L2 Combination", "Mean CV RMSE ($)": 15451.56, "Std RMSE ($)": 305.52, "Holdout R²": 0.8866, "Holdout MAE ($)": 11927.45, "Time (s)": 0.09},
    ])
    
    st.dataframe(
        benchmark_df.style.format({
            "Mean CV RMSE ($)": "${:,.2f}",
            "Std RMSE ($)": "${:,.2f}",
            "Holdout R²": "{:.4f}",
            "Holdout MAE ($)": "${:,.2f}",
            "Time (s)": "{:.2f}s"
        }),
        use_container_width=True
    )
    
    st.bar_chart(data=benchmark_df.set_index("Algorithm")["Mean CV RMSE ($)"])

with tab_weights:
    st.markdown("### ⚖️ Top Learned Regression Coefficients")
    st.markdown("The regularized linear pipeline learns interpretable, dollar-denominated impacts for each job attribute:")
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("#### 🟢 Top Positive Compensation Drivers")
        pos_drivers = pd.DataFrame([
            {"Feature Attribute": "Executive Level", "Impact on Salary": "+$50,126"},
            {"Feature Attribute": "United States Location", "Impact on Salary": "+$35,457"},
            {"Feature Attribute": "Large Enterprise (>250 emp)", "Impact on Salary": "+$17,903"},
            {"Feature Attribute": "Deep Learning Engineer", "Impact on Salary": "+$12,650"},
            {"Feature Attribute": "Singapore Location", "Impact on Salary": "+$9,896"},
            {"Feature Attribute": "AI Researcher", "Impact on Salary": "+$9,741"},
            {"Feature Attribute": "Australia Location", "Impact on Salary": "+$8,521"},
            {"Feature Attribute": "Senior Experience", "Impact on Salary": "+$7,541"},
        ])
        st.dataframe(pos_drivers, use_container_width=True)
        
    with col_w2:
        st.markdown("#### 🔴 Top Downward Adjustment Drivers")
        neg_drivers = pd.DataFrame([
            {"Feature Attribute": "Entry Level", "Impact on Salary": "-$54,829"},
            {"Feature Attribute": "India Location", "Impact on Salary": "-$38,512"},
            {"Feature Attribute": "Mid Level", "Impact on Salary": "-$26,254"},
            {"Feature Attribute": "Data Analyst", "Impact on Salary": "-$26,278"},
            {"Feature Attribute": "BI Analyst", "Impact on Salary": "-$23,814"},
            {"Feature Attribute": "Spain Location", "Impact on Salary": "-$12,036"},
            {"Feature Attribute": "Small Startup (<50 emp)", "Impact on Salary": "-$10,046"},
            {"Feature Attribute": "Data Engineer", "Impact on Salary": "-$5,810"},
        ])
        st.dataframe(neg_drivers, use_container_width=True)

with tab_dataset:
    st.markdown("### 📁 Dataset Overview (`ai_ds_salaries_DUMMY.csv`)")
    st.markdown("""
    - **Total Records:** 8,000 instances
    - **Features:** 8 predictor variables (2 numeric, 6 categorical)
    - **Target Variable:** `salary_in_usd`
    - **Average Salary:** $146,646 | **Std Dev:** $44,898
    - **Minimum:** $25,000 | **Maximum:** $300,400
    """)
    
    st.info("The dataset includes missing values in `employee_residence` (160 rows) and `remote_ratio` (120 rows) which are imputed dynamically inside the scikit-learn pipeline using median and mode imputation.")
