"""Technical Details - Deep dive into model architecture and performance."""
import streamlit as st
import sys
import os
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui_components.styling import apply_custom_css, create_section_header
from src.visualization.model_viz import (
    create_confusion_matrix,
    create_roc_curve,
    create_precision_recall_curve,
    create_classification_metrics_table
)
from src.visualization.feature_viz import (
    create_feature_importance_chart,
    create_feature_categories_pie,
    get_feature_category_summary
)
from src.visualization.comparison_viz import (
    create_model_comparison_bars,
    create_model_comparison_table,
    create_model_radar_chart,
    create_accuracy_vs_time_scatter
)

# Apply custom styling
apply_custom_css()

# Page Header
st.markdown('<h1 class="main-title">🔬 Technical Details</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Comprehensive Model Architecture & Performance Analysis</p>', unsafe_allow_html=True)

st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "🏗️ Architecture & Training",
    "📈 Performance Metrics",
    "🔍 Feature Analysis",
    "📊 Model Comparison"
])

# ============================================================================
# TAB 1: ARCHITECTURE & TRAINING
# ============================================================================
with tab1:
    st.markdown("## Model Architecture & Training Process")

    col1, col2 = st.columns([1, 1])

    with col1:
        create_section_header("Model Specification")
        st.markdown("""
        **Algorithm**: XGBoost Classifier
        **Type**: Binary Classification
        **Target**: High School Diploma (0=No, 1=Yes)
        **Training Date**: October 2023

        ### Hyperparameters

        | Parameter | Value |
        |-----------|-------|
        | `colsample_bytree` | 0.7056 |
        | `gamma` | 6.5289 |
        | `max_depth` | 11 |
        | `min_child_weight` | 7.0 |
        | `reg_alpha` | 40.0 |
        | `reg_lambda` | 0.0067 |
        | `random_state` | 42 |

        **Optimization Method**:
        - Bayesian Optimization (Hyperopt)
        - 100 evaluation rounds
        - 10-fold Stratified Cross-Validation
        """)

    with col2:
        create_section_header("Data Pipeline")
        st.markdown("""
        **Dataset Statistics**:
        - Total Students: 52,942
        - Training Set: 42,353 (80%)
        - Test Set: 10,589 (20%)
        - Class Distribution: 81.2% graduated, 18.8% at-risk

        ### Preprocessing Steps

        **1. Numeric Features**:
        ```
        SimpleImputer (median strategy)
        ↓
        StandardScaler (normalization)
        ```

        **2. Categorical Features**:
        ```
        SimpleImputer (most_frequent)
        ↓
        OneHotEncoder (handle_unknown='ignore')
        ```

        **3. Feature Engineering**:
        - 15 raw input features
        - 28 features after encoding
        - No feature selection applied

        **4. Validation Strategy**:
        - 10-fold Stratified Cross-Validation
        - Stratification preserves class balance
        """)

    st.markdown("---")

    # Training Process
    create_section_header("Training Process")

    process_col1, process_col2, process_col3 = st.columns(3)

    with process_col1:
        st.markdown("""
        ### 1️⃣ Data Preparation
        - Load raw CSV data
        - Train/test split (80/20)
        - Stratified sampling
        - Feature validation
        """)

    with process_col2:
        st.markdown("""
        ### 2️⃣ Hyperparameter Tuning
        - Bayesian optimization
        - 100 iterations
        - Cross-validation scoring
        - Best parameters selected
        """)

    with process_col3:
        st.markdown("""
        ### 3️⃣ Model Training
        - Train on full training set
        - Evaluate on held-out test set
        - Generate performance metrics
        - Save artifacts
        """)

    st.markdown("---")

    # Model Characteristics
    create_section_header("Model Characteristics")

    st.markdown("""
    ### Strengths
    - ✅ High accuracy (90.6%) across all metrics
    - ✅ Excellent precision (94.7%) - low false positives
    - ✅ Strong recall (93.9%) - catches most at-risk students
    - ✅ Robust to class imbalance (81.2% vs 18.8%)
    - ✅ Interpretable feature importance

    ### Considerations
    - ⚠️ Training time: ~45 seconds (acceptable for batch predictions)
    - ⚠️ Model size: ~327 KB (easily deployable)
    - ⚠️ Requires full feature set for predictions
    - ⚠️ Best performance with complete data (minimal missing values)
    """)

# ============================================================================
# TAB 2: PERFORMANCE METRICS
# ============================================================================
with tab2:
    st.markdown("## Performance Metrics & Evaluation")

    # Load test predictions
    try:
        with open('artifacts/test_predictions.json', 'r') as f:
            pred_data = json.load(f)

        y_test = np.array(pred_data['y_test'])
        y_pred = np.array(pred_data['y_pred'])
        y_proba = np.array(pred_data['y_proba'])

        # Confusion Matrix
        create_section_header("Confusion Matrix")
        st.markdown("""
        The confusion matrix shows how well the model distinguishes between students
        who will graduate vs. those at risk of not graduating.
        """)

        fig_cm = create_confusion_matrix(y_test, y_pred)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")

        # ROC and PR Curves
        create_section_header("Performance Curves")

        curve_col1, curve_col2 = st.columns(2)

        with curve_col1:
            st.markdown("### ROC Curve")
            st.markdown("""
            **ROC (Receiver Operating Characteristic)**
            Shows the trade-off between True Positive Rate and False Positive Rate across different thresholds.
            """)

            fig_roc = create_roc_curve(y_test, y_proba)
            st.plotly_chart(fig_roc, use_container_width=True)

        with curve_col2:
            st.markdown("### Precision-Recall Curve")
            st.markdown("""
            **Precision-Recall Curve**
            Shows the trade-off between precision and recall. The red star marks our decision threshold (0.5).
            """)

            fig_pr = create_precision_recall_curve(y_test, y_proba)
            st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("---")

        # Classification Metrics Table
        create_section_header("Classification Metrics")

        metrics_df = create_classification_metrics_table(y_test, y_pred)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("""
            ### Metric Definitions

            **Accuracy**: Proportion of correct predictions (both classes)
            _Formula_: (TP + TN) / (TP + TN + FP + FN)

            **Precision**: When we predict "will graduate", how often are we correct?
            _Formula_: TP / (TP + FP)

            **Recall**: Of all students who actually graduate, how many do we identify?
            _Formula_: TP / (TP + FN)

            **F1-Score**: Harmonic mean of precision and recall
            _Formula_: 2 × (Precision × Recall) / (Precision + Recall)
            """)

    except FileNotFoundError:
        st.warning("⚠️ Test predictions file not found. Run `python scripts/prepare_artifacts.py` to generate predictions.")

    except Exception as e:
        st.error(f"Error loading performance metrics: {e}")

# ============================================================================
# TAB 3: FEATURE ANALYSIS
# ============================================================================
with tab3:
    st.markdown("## Feature Importance Analysis")

    try:
        # Feature Importance Chart
        create_section_header("Top 15 Most Important Features")

        st.markdown("""
        Feature importance scores indicate how much each feature contributes to the model's predictions.
        Higher scores mean the feature has more influence on graduation predictions.
        """)

        fig_importance = create_feature_importance_chart(top_n=15)
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("---")

        # Feature Categories
        create_section_header("Feature Categories")

        cat_col1, cat_col2 = st.columns([1, 1])

        with cat_col1:
            fig_pie = create_feature_categories_pie()
            st.plotly_chart(fig_pie, use_container_width=True)

        with cat_col2:
            category_summary = get_feature_category_summary()

            st.markdown("### Category Breakdown")

            for _, row in category_summary.iterrows():
                st.markdown(f"**{row['category']}**: {row['percentage_str']}")

            st.markdown("---")

            st.markdown("""
            ### Key Insights

            1. **Standardized Tests dominate** (56.3%): ACT scores are the strongest predictors
            2. **Academic Performance** (30.4%): GPA and course scores are critical
            3. **Demographics** (6.5%): Background factors play a smaller but notable role
            4. **Support Services** (6.8%): Need for IEP/ELL services indicates risk

            **Takeaway**: Academic readiness metrics (tests + GPA) account for **86.7% of predictive power**.
            """)

        st.markdown("---")

        # Feature Definitions
        create_section_header("Feature Definitions")

        feature_definitions = pd.DataFrame({
            'Feature': [
                'scale_score_11_eng', 'scale_score_11_comp', 'gpa',
                'scale_score_11_read', 'scale_score_11_math',
                'read_ss', 'math_ss', 'pct_days_absent',
                'ap_ever_take_class', 'race_ethnicity', 'male',
                'frpl', 'iep', 'ell', 'ever_alternative'
            ],
            'Category': [
                'Standardized Tests', 'Standardized Tests', 'Academic Performance',
                'Standardized Tests', 'Standardized Tests',
                'Academic Performance', 'Academic Performance', 'Attendance',
                'Academic Performance', 'Demographics', 'Demographics',
                'Demographics', 'Support Services', 'Support Services', 'Support Services'
            ],
            'Description': [
                '11th grade ACT English score',
                '11th grade ACT Composite score',
                'Cumulative GPA (0.0-4.0 scale)',
                '11th grade ACT Reading score',
                '11th grade ACT Math score',
                'Standardized reading score',
                'Standardized math score',
                'Percentage of school days absent',
                'Has taken AP courses (yes/no)',
                'Racial/ethnic background',
                'Gender (yes=male, no=female)',
                'Free/Reduced Price Lunch status',
                'Individualized Education Program',
                'English Language Learner status',
                'Alternative school enrollment history'
            ]
        })

        st.dataframe(feature_definitions, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading feature analysis: {e}")

# ============================================================================
# TAB 4: MODEL COMPARISON
# ============================================================================
with tab4:
    st.markdown("## Model Comparison")

    try:
        # Comparison Table
        create_section_header("Comprehensive Model Comparison")

        st.markdown("""
        We evaluated **7 different algorithms** plus a baseline to identify the best performer:
        """)

        df_comparison = create_model_comparison_table()
        st.dataframe(
            df_comparison.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], axis=0),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Model Comparison Chart
        create_section_header("Performance Visualization")

        fig_comp = create_model_comparison_bars(metric='all')
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # Radar Chart
        create_section_header("Top 3 Models - Multidimensional Comparison")

        st.markdown("""
        Radar chart comparing the top 3 models across all performance metrics:
        """)

        fig_radar = create_model_radar_chart(['xgboost', 'random_forest', 'knn'])
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # Accuracy vs Time
        create_section_header("Accuracy vs Training Time Trade-off")

        st.markdown("""
        This scatter plot shows the relationship between model accuracy and training time.
        Models in the upper-left are ideal (high accuracy, low training time).
        """)

        fig_scatter = create_accuracy_vs_time_scatter()
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # Model Selection Rationale
        create_section_header("Model Selection Rationale")

        st.markdown("""
        ### Why XGBoost?

        **Advantages**:
        - ✅ **Highest Accuracy** (90.6%) among all models tested
        - ✅ **Excellent Balance** between precision (94.7%) and recall (93.9%)
        - ✅ **Robust Performance** across cross-validation folds
        - ✅ **Feature Importance** provides interpretability
        - ✅ **Handles Imbalance** well without special techniques

        **Trade-offs**:
        - ⚠️ **Training Time** (45.2s) - longer than simpler models but acceptable for batch processing
        - ⚠️ **Complexity** - more parameters than linear models
        - ⚠️ **Resource Usage** - requires more memory than naive approaches

        **Alternative Considerations**:
        - **Random Forest** (90.5% accuracy) was a close second and could be used if faster training is needed
        - **KNN** (87.3% accuracy) offers fastest training (2.1s) but lower performance
        - **Logistic Regression** (85.6% accuracy) provides simplest interpretability but lower accuracy

        ### Conclusion
        XGBoost was selected because it provides the **best overall performance** for this critical application
        where **accuracy matters more than training speed**. The small additional training time (vs Random Forest)
        is justified by the **0.1% accuracy improvement**.
        """)

    except Exception as e:
        st.error(f"Error loading model comparison: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>For more details, see the <strong>Model Cards</strong> page for comprehensive documentation.</p>
</div>
""", unsafe_allow_html=True)
