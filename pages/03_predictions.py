"""Interactive Predictions - Student Graduation Risk Assessment."""
import streamlit as st
import sys
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui_components.styling import apply_custom_css
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import engineer_features

# Apply custom styling
apply_custom_css()


@st.cache_resource
def load_model_and_preprocessor():
    """Load model and preprocessor with caching."""
    model_path = os.path.join("artifacts", "model.pkl")
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


def create_gauge_chart(probability, title="Graduation Probability"):
    """Create an interactive gauge chart for probability display."""
    prob_percent = probability * 100

    # Determine color based on probability
    if prob_percent >= 75:
        color = "#28a745"  # Green
    elif prob_percent >= 50:
        color = "#ffc107"  # Yellow
    else:
        color = "#dc3545"  # Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_percent,
        title={'text': title, 'font': {'size': 24, 'color': '#333'}},
        delta={'reference': 75, 'increasing': {'color': "#28a745"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffe6e6'},
                {'range': [50, 75], 'color': '#fff9e6'},
                {'range': [75, 100], 'color': '#e6ffe6'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        },
        number={'suffix': "%", 'font': {'size': 40}}
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"}
    )

    return fig


def create_risk_indicator_chart(risk_factors):
    """Create a horizontal bar chart showing risk indicators."""
    factors = list(risk_factors.keys())
    scores = list(risk_factors.values())

    # Create color scale based on risk level
    colors = ['#28a745' if s > 70 else '#ffc107' if s > 40 else '#dc3545' for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=factors,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f"{s:.1f}%" for s in scores],
        textposition='auto',
    ))

    fig.update_layout(
        title="Student Risk Factor Analysis",
        xaxis_title="Risk Score (%)",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.02)",
        font=dict(size=12, color='#333'),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )

    return fig


def calculate_risk_factors(user_input):
    """Calculate individual risk factor scores based on student inputs."""
    risk_scores = {}

    # Academic Performance (GPA)
    gpa = user_input['gpa']
    risk_scores['Academic (GPA)'] = min(100, (gpa / 4.0) * 100)

    # Attendance
    attendance = 100 - user_input['pct_days_absent']
    risk_scores['Attendance'] = max(0, attendance)

    # Standardized Test Performance
    act_avg = (user_input['scale_score_11_comp'] + user_input['scale_score_11_eng'] +
               user_input['scale_score_11_math'] + user_input['scale_score_11_read']) / 4
    risk_scores['ACT Performance'] = min(100, (act_avg / 36) * 100)

    # Course Performance
    course_avg = (user_input['math_ss'] + user_input['read_ss']) / 2
    risk_scores['Course Scores'] = min(100, (course_avg / 100) * 100)

    # Support Services (inverse - fewer services needed is better)
    support_count = sum([
        1 if user_input['iep'] == 'yes' else 0,
        1 if user_input['ell'] == 'yes' else 0,
        1 if user_input['ever_alternative'] == 'yes' else 0
    ])
    risk_scores['Support Services'] = max(0, 100 - (support_count * 25))

    # Advanced Coursework
    risk_scores['AP Participation'] = 100 if user_input['ap_ever_take_class'] == 'yes' else 30

    return risk_scores


def get_user_input_form():
    """Create organized sidebar form for user input."""
    st.sidebar.title("🎓 Student Profile")
    st.sidebar.markdown("---")

    # Demographics Section
    with st.sidebar.expander("👤 Demographics", expanded=True):
        male = st.selectbox(
            "Gender (Male)",
            ("yes", "no"),
            help="Select student gender"
        )
        race_ethnicity = st.selectbox(
            "Race/Ethnicity",
            ('White', 'African-American', 'Asian/Pacific Islander', 'Hispanic', 'Multiple/Native American'),
            help="Select student race/ethnicity"
        )
        frpl = st.selectbox(
            "Free/Reduced Price Lunch",
            ('yes', 'no'),
            help="Qualifies for FRPL"
        )

    # Support Services Section
    with st.sidebar.expander("🤝 Support Services"):
        iep = st.selectbox(
            "IEP (Individualized Education Program)",
            ('no', 'yes'),
            help="Receives special education services"
        )
        ell = st.selectbox(
            "ELL (English Language Learner)",
            ('no', 'yes'),
            help="Receives ELL services"
        )
        ever_alternative = st.selectbox(
            "Alternative School Enrollment",
            ('no', 'yes'),
            help="Ever enrolled in alternative school"
        )

    # Academic Performance Section
    with st.sidebar.expander("📚 Academic Performance", expanded=True):
        gpa = st.slider(
            'GPA',
            0.0, 4.0, 2.80, 0.01,
            help="Cumulative GPA (0.0 - 4.0)"
        )
        ap_ever_take_class = st.selectbox(
            "AP Course Participation",
            ('no', 'yes'),
            help="Has taken AP courses"
        )

        col1, col2 = st.columns(2)
        with col1:
            math_ss = st.number_input(
                'Math Score',
                0, 200, 50,
                help="Standardized math score"
            )
        with col2:
            read_ss = st.number_input(
                'Reading Score',
                0, 200, 50,
                help="Standardized reading score"
            )

    # Attendance Section
    with st.sidebar.expander("📅 Attendance"):
        pct_days_absent = st.slider(
            'Days Absent (%)',
            0.0, 100.0, 8.5, 0.5,
            help="Percentage of school days missed"
        )
        st.info(f"Attendance Rate: {100 - pct_days_absent:.1f}%")

    # ACT Scores Section
    with st.sidebar.expander("🎯 ACT Scores"):
        scale_score_11_comp = st.slider(
            'Composite Score',
            0.0, 36.0, 19.0, 0.5,
            help="ACT Composite Score"
        )
        scale_score_11_eng = st.slider(
            'English Score',
            0.0, 36.0, 19.0, 0.5,
            help="ACT English Score"
        )
        scale_score_11_math = st.slider(
            'Math Score',
            0.0, 36.0, 19.0, 0.5,
            help="ACT Math Score"
        )
        scale_score_11_read = st.slider(
            'Reading Score',
            0.0, 36.0, 20.0, 0.5,
            help="ACT Reading Score"
        )

        avg_act = (scale_score_11_comp + scale_score_11_eng +
                   scale_score_11_math + scale_score_11_read) / 4
        st.metric("Average ACT", f"{avg_act:.1f}")

    features = {
        'male': male,
        'race_ethnicity': race_ethnicity,
        'iep': iep,
        'frpl': frpl,
        'ell': ell,
        'ap_ever_take_class': ap_ever_take_class,
        'ever_alternative': ever_alternative,
        'gpa': gpa,
        'pct_days_absent': pct_days_absent,
        'math_ss': math_ss,
        'read_ss': read_ss,
        'scale_score_11_comp': scale_score_11_comp,
        'scale_score_11_eng': scale_score_11_eng,
        'scale_score_11_math': scale_score_11_math,
        'scale_score_11_read': scale_score_11_read
    }

    return pd.DataFrame([features]), features


# Page Header
st.markdown('<h1 class="main-title">🎯 Make Predictions</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Interactive Student Graduation Risk Assessment</p>', unsafe_allow_html=True)

st.markdown("---")

# Info box
st.markdown("""
<div class="info-box">
    <h4>👉 How to Use This Tool</h4>
    <ol>
        <li>Enter student information in the <strong>sidebar</strong> (left panel)</li>
        <li>Adjust sliders and dropdowns to match the student's profile</li>
        <li>Click <strong>🚀 Run Prediction Model</strong> to generate risk assessment</li>
        <li>Review results, risk factors, and personalized recommendations</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Get user input
user_input_df, user_input_dict = get_user_input_form()

# Prediction button
st.markdown("### 🔮 Generate Prediction")

predict_button = st.button("🚀 Run Prediction Model", use_container_width=True)

if predict_button or 'prediction_made' in st.session_state:
    st.session_state.prediction_made = True

    # Show progress
    with st.spinner('🔄 Analyzing student profile...'):
        try:
            # Make prediction
            predict_pipeline = PredictPipeline()
            prediction, probability = predict_pipeline.predict(user_input_df)

            # Store in session state
            st.session_state.prediction = prediction[0]
            st.session_state.probability = probability[0]

            # Success message
            st.success("✅ Prediction completed successfully!")

            # Results section
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")

            # Main prediction display
            col1, col2 = st.columns([1, 1])

            with col1:
                # Prediction outcome
                if prediction[0] == 1:
                    st.markdown("""
                    <div class="success-box">
                        <h2 style="color: #28a745; margin: 0;">✅ DIPLOMA PREDICTED</h2>
                        <p style="font-size: 1.2rem; margin-top: 10px;">
                            Student is predicted to graduate with a high school diploma
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h2 style="color: #dc3545; margin: 0;">⚠️ AT RISK</h2>
                        <p style="font-size: 1.2rem; margin-top: 10px;">
                            Student is at risk of not graduating - intervention recommended
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Key metrics
                st.markdown("### 📈 Key Metrics")
                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Graduation Probability",
                        f"{probability[0] * 100:.1f}%",
                        delta=f"{(probability[0] * 100 - 75):.1f}% vs threshold"
                    )

                with metric_col2:
                    risk_level = "Low" if probability[0] > 0.75 else "Medium" if probability[0] > 0.5 else "High"
                    st.metric("Risk Level", risk_level)

            with col2:
                # Gauge chart
                gauge_fig = create_gauge_chart(probability[0], "Graduation Probability")
                st.plotly_chart(gauge_fig, use_container_width=True)

            # Risk factors analysis
            st.markdown("---")
            st.markdown("### 🎯 Risk Factor Analysis")

            risk_factors = calculate_risk_factors(user_input_dict)
            risk_chart = create_risk_indicator_chart(risk_factors)
            st.plotly_chart(risk_chart, use_container_width=True)

            # Detailed student profile
            st.markdown("---")
            st.markdown("### 👤 Student Profile Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Academic Performance**")
                st.write(f"• GPA: {user_input_dict['gpa']:.2f}")
                st.write(f"• Math Score: {user_input_dict['math_ss']}")
                st.write(f"• Reading Score: {user_input_dict['read_ss']}")
                st.write(f"• AP Courses: {user_input_dict['ap_ever_take_class'].title()}")

            with col2:
                st.markdown("**ACT Scores**")
                st.write(f"• Composite: {user_input_dict['scale_score_11_comp']:.1f}")
                st.write(f"• English: {user_input_dict['scale_score_11_eng']:.1f}")
                st.write(f"• Math: {user_input_dict['scale_score_11_math']:.1f}")
                st.write(f"• Reading: {user_input_dict['scale_score_11_read']:.1f}")

            with col3:
                st.markdown("**Attendance & Support**")
                engineered_input = engineer_features(user_input_df)
                st.write(f"• Attendance Rate (model feature): {engineered_input['attendance_rate'].iloc[0]:.1f}%")
                st.write(f"• IEP Services: {user_input_dict['iep'].title()}")
                st.write(f"• ELL Services: {user_input_dict['ell'].title()}")
                st.write(f"• FRPL: {user_input_dict['frpl'].title()}")

            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommended Actions")

            if prediction[0] == 0:
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Intervention Strategies:</h4>
                    <ul>
                        <li><strong>Academic Support:</strong> Consider tutoring or academic coaching</li>
                        <li><strong>Attendance Monitoring:</strong> Implement attendance intervention plan</li>
                        <li><strong>Counseling:</strong> Schedule meeting with guidance counselor</li>
                        <li><strong>Parent Engagement:</strong> Increase family communication</li>
                        <li><strong>Progress Monitoring:</strong> Weekly check-ins on academic progress</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Maintenance Strategies:</h4>
                    <ul>
                        <li><strong>Continue Current Support:</strong> Maintain existing support services</li>
                        <li><strong>College Preparation:</strong> Begin college and career planning</li>
                        <li><strong>Advanced Coursework:</strong> Encourage AP/honors course enrollment</li>
                        <li><strong>Scholarship Planning:</strong> Explore scholarship opportunities</li>
                        <li><strong>Mentorship:</strong> Connect with peer or adult mentors</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Export functionality
            st.markdown("---")
            st.markdown("### 📥 Export Prediction")

            report = {
                "student_profile": user_input_dict,
                "prediction": "Diploma" if prediction[0] == 1 else "At Risk",
                "probability": float(probability[0]),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "timestamp": datetime.now().isoformat()
            }

            st.download_button(
                label="📥 Download Prediction Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"student_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.exception(e)
