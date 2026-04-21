"""Executive Summary - Main landing page for stakeholders."""
import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui_components.styling import apply_custom_css, create_hero_section, create_metric_card, create_section_header
from src.visualization.comparison_viz import create_model_comparison_bars, load_model_performance
from src.visualization.feature_viz import get_feature_importance_data, get_feature_category_summary
import plotly.express as px

# Apply custom styling
apply_custom_css()

# Hero Section
st.markdown('<h1 class="hero-title">🎓 POC Early Warning System</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">ML-Powered Student Graduation Risk Prediction</p>', unsafe_allow_html=True)

# Key Achievement Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "90.6%", "+9.3% vs baseline")

with col2:
    st.metric("Precision", "94.7%", "High confidence")

with col3:
    st.metric("Students Analyzed", "42,353", "Training dataset")

with col4:
    st.metric("Models Evaluated", "7", "XGBoost selected")

st.markdown("---")

# Business Value Proposition
st.markdown("## 🎯 Business Value Proposition")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("""
    ### The Problem

    Traditional early warning systems use a **one-size-fits-all approach** with generic indicators:

    - ❌ Attendance below 90%
    - ❌ Number of D's and F's
    - ❌ Suspension frequency

    These methods **lack specificity** about what students are at risk of—graduation, college admission, or standardized test success.
    """)

with col_right:
    st.markdown("""
    ### Our Solution

    This system leverages **machine learning** to create **individualized risk indicators**:

    - ✅ **90.6% accuracy** in graduation prediction
    - ✅ **Personalized risk assessment** for each student
    - ✅ **Early intervention** targeting
    - ✅ **Actionable insights** for counselors

    The model analyzes **15 predictive features** across demographics, academics, attendance, and standardized tests.
    """)

st.markdown("---")

# Model Performance Section
create_section_header("📊 Model Performance")

st.markdown("""
We evaluated **7 different machine learning algorithms** to identify the best performer:
""")

# Model comparison chart
try:
    fig = create_model_comparison_bars(metric='all')
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ **XGBoost Classifier selected as best performer**: 90.6% accuracy, 94.7% precision, 93.9% recall")

except Exception as e:
    st.error(f"Could not load model comparison: {e}")

# Performance highlights
st.markdown("### Key Performance Highlights")

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.markdown("""
    **Accuracy: 90.6%**
    - Correctly identifies graduation outcome for 9 out of 10 students
    - Significantly outperforms baseline (81.3%)
    """)

with perf_col2:
    st.markdown("""
    **Precision: 94.7%**
    - When predicting "will graduate", correct 95% of the time
    - High confidence in positive predictions
    """)

with perf_col3:
    st.markdown("""
    **Recall: 93.9%**
    - Identifies 94% of students who will graduate
    - Minimal false negatives
    """)

st.markdown("---")

# Top Predictive Features
create_section_header("🔑 Top Predictive Features")

st.markdown("""
The model identified the most important indicators for graduation prediction:
""")

try:
    # Get top 5 features
    df_features = get_feature_importance_data().head(5)

    fig = px.bar(
        df_features,
        x='importance',
        y='feature',
        orientation='h',
        text=df_features['importance'].apply(lambda x: f'{x*100:.1f}%'),
        color='importance',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        title='Top 5 Most Important Features',
        xaxis_title='Importance Score',
        yaxis_title='',
        height=400,
        showlegend=False,
        font=dict(size=12)
    )

    fig.update_traces(textposition='auto')

    st.plotly_chart(fig, use_container_width=True)

    # Feature explanations
    st.markdown("""
    **Key Predictive Indicators:**

    1. **ACT English Score** (30.4%) - Strongest single predictor of graduation success
    2. **ACT Composite Score** (15.5%) - Overall standardized test performance
    3. **GPA** (9.2%) - Cumulative academic achievement
    4. **ACT Reading/Math Scores** - Additional standardized test metrics

    **Insight**: Standardized test performance accounts for over **56% of predictive power**,
    highlighting the importance of academic readiness assessment.
    """)

except Exception as e:
    st.error(f"Could not load feature importance: {e}")

st.markdown("---")

# Feature Categories
st.markdown("### Feature Categories Breakdown")

try:
    category_summary = get_feature_category_summary()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pie chart
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Pie(
            labels=category_summary['category'],
            values=category_summary['importance'],
            hole=0.4,
            textinfo='label+percent'
        )])

        fig.update_layout(
            title='Importance by Feature Category',
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Category Contributions:**")
        for _, row in category_summary.iterrows():
            st.markdown(f"- **{row['category']}**: {row['percentage_str']}")

        st.markdown("""
        ---

        **Key Takeaway**: Academic performance metrics (tests + GPA)
        combine for **86.7% of predictive power**, emphasizing the
        critical role of academic preparation in graduation outcomes.
        """)

except Exception as e:
    st.warning("Feature category analysis unavailable")

st.markdown("---")

# Use Cases
create_section_header("💡 Use Cases & Applications")

use_col1, use_col2, use_col3 = st.columns(3)

with use_col1:
    st.markdown("""
    ### 🎯 Early Intervention

    - Identify at-risk students early
    - Target support resources efficiently
    - Monitor intervention effectiveness
    - Reduce dropout rates
    """)

with use_col2:
    st.markdown("""
    ### 📊 Resource Allocation

    - Prioritize counseling resources
    - Plan tutoring programs
    - Allocate support staff
    - Budget for interventions
    """)

with use_col3:
    st.markdown("""
    ### 📈 Performance Monitoring

    - Track student progress over time
    - Measure program effectiveness
    - Identify systemic issues
    - Report to stakeholders
    """)

st.markdown("---")

# Next Steps / Navigation
create_section_header("🚀 Explore Further")

st.markdown("""
Dive deeper into the technical implementation, try live predictions, or review comprehensive model documentation:
""")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    <div style="background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 10px;">🔬</div>
        <h3>Technical Details</h3>
        <p>Explore model architecture, performance metrics, and training process</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
    <div style="background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 10px;">🎯</div>
        <h3>Try Predictions</h3>
        <p>Interactive demo with student risk assessment and recommendations</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown("""
    <div style="background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 10px;">📋</div>
        <h3>Documentation</h3>
        <p>Comprehensive model card, fairness analysis, and quality reports</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px; margin-top: 30px;">
    <p><strong>POC Early Warning System</strong> | ML-Powered Graduation Prediction</p>
    <p>90.6% Accuracy • 42,353 Students • 7 Models Evaluated</p>
</div>
""", unsafe_allow_html=True)
