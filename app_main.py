"""
POC Early Warning System - Main Application
Unified multi-page Streamlit application for comprehensive ML POC presentation.
"""
import streamlit as st
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ui_components.styling import apply_custom_css

# Page configuration
st.set_page_config(
    page_title="POC Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/rjwdata/poc-early-warning',
        'Report a bug': 'https://github.com/rjwdata/poc-early-warning/issues',
        'About': """
        # POC Early Warning System

        ML-Powered Student Graduation Risk Prediction

        **Version**: 1.0
        **Accuracy**: 90.6%
        **Model**: XGBoost Classifier

        This application predicts high school graduation outcomes using machine learning
        to enable early intervention and support for at-risk students.
        """
    }
)

# Apply custom styling
apply_custom_css()

# Define navigation structure
pages = {
    "🏠 Overview": [
        st.Page("pages/01_executive_summary.py", title="Executive Summary", icon="🏠", default=True)
    ],
    "Technical Documentation": [
        st.Page("pages/02_technical_details.py", title="Technical Details", icon="🔬"),
        st.Page("pages/04_model_cards.py", title="Model Cards", icon="📋"),
        st.Page("pages/05_data_quality.py", title="Data Quality", icon="📊")
    ],
    "Applications": [
        st.Page("pages/03_predictions.py", title="Make Predictions", icon="🎯")
    ]
}

# Create navigation
pg = st.navigation(pages)

# Run the selected page
pg.run()

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📌 Quick Links")
    st.markdown("""
    - [Executive Summary](/Executive_Summary) - Business overview
    - [Technical Details](/Technical_Details) - Deep dive
    - [Make Predictions](/Make_Predictions) - Interactive demo
    - [Model Cards](/Model_Cards) - Documentation
    - [Data Quality](/Data_Quality) - Quality reports
    """)

    st.markdown("---")
    st.markdown("### 📊 System Overview")
    st.markdown("""
    **Model**: XGBoost Classifier
    **Accuracy**: 90.6%
    **Precision**: 94.7%
    **Recall**: 93.9%
    **Training Data**: 42,353 students
    **Features**: 15 indicators
    """)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **POC Early Warning System**

    Version 1.0
    October 2023

    Machine learning system for predicting high school graduation risk.

    [GitHub Repository](https://github.com/rjwdata/poc-early-warning)
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem;">
        <p>Made with ❤️ by hawkeye</p>
        <p>© 2023 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)
