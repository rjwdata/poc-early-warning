"""
POC Early Warning System - Student Diploma Prediction Application
Modern Streamlit interface with enhanced visualizations and UX
"""
import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import load_object


# Page configuration
st.set_page_config(
    page_title="POC Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
    }

    /* Info box styling */
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }

    /* Success box */
    .success-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 20px 0;
    }

    /* Warning box */
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 20px 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #145a8c;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """
    Load model and preprocessor with caching for improved performance.
    Uses st.cache_resource to cache the loaded models across reruns.

    Returns:
        tuple: (model, preprocessor)
    """
    transformer_path = os.path.join("artifacts", "preprocessor.pkl")
    model_path = os.path.join("artifacts", "model.pkl")

    model = joblib.load(model_path)
    preprocessor = joblib.load(transformer_path)

    return model, preprocessor


@st.cache_data
def load_training_data():
    """Load training data for overview with caching."""
    try:
        df = pd.read_csv(os.path.join('data', 'raw', 'train', 'train.csv'))
        return df
    except FileNotFoundError:
        return None


def create_gauge_chart(probability, title="Graduation Probability"):
    """
    Create an interactive gauge chart for probability display.

    Args:
        probability: Probability value between 0 and 1
        title: Chart title

    Returns:
        plotly figure object
    """
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


def create_risk_indicator_chart(features_df, risk_factors):
    """
    Create a horizontal bar chart showing risk indicators.

    Args:
        features_df: DataFrame with student features
        risk_factors: Dictionary of risk factor scores

    Returns:
        plotly figure object
    """
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
    """
    Calculate individual risk factor scores based on student inputs.

    Args:
        user_input: Dictionary of student features

    Returns:
        Dictionary of risk factor scores
    """
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
    """
    Create organized sidebar form for user input with sections.

    Returns:
        DataFrame with user inputs
    """
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


def main():
    """Main application function."""

    # Header
    st.markdown('<h1 class="main-title">🎓 POC Early Warning System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Individualized Student Graduation Risk Prediction</p>', unsafe_allow_html=True)

    # Load models
    try:
        model, preprocessor = load_model_and_preprocessor()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

    # Introduction tabs
    tab1, tab2, tab3 = st.tabs(["📊 Make Prediction", "📈 About the System", "📚 Data Overview"])

    with tab2:
        st.markdown("""
        ### 🎯 Purpose

        Traditional early warning systems use a **one-size-fits-all approach**, monitoring generic indicators such as:
        - Attendance below 90%
        - Number of D's and F's
        - Suspension frequency

        However, these methods lack specificity about **what students are at risk of**—whether it's graduation,
        college admission, or success on placement tests.

        ### ✨ Our Solution

        This system leverages **machine learning** to create **individualized risk indicators** that predict whether
        a student is at risk of not graduating from high school. The model considers:

        - **Demographics**: Gender, race/ethnicity, FRPL status
        - **Academic Performance**: GPA, course scores, AP participation
        - **Attendance**: Days missed throughout the year
        - **Standardized Tests**: ACT scores across all subjects
        - **Support Services**: IEP, ELL, alternative school history

        ### 🔬 Model Performance

        - **Algorithm**: XGBoost Classifier
        - **Accuracy**: 90.6%
        - **Training Data**: 42,353 students
        - **Features**: 15 predictive indicators

        ### 📖 How to Use

        1. Enter student information in the **sidebar** (left panel)
        2. Adjust sliders and dropdowns to match the student's profile
        3. View real-time **prediction results** in the main panel
        4. Analyze **risk factors** to understand areas of concern
        5. Use insights for **intervention planning**

        ### 🔗 Resources

        [![GitHub](https://img.shields.io/badge/POC%20Early%20Warning-GitHub-100000?logo=github&logoColor=white)](https://github.com/rjwdata/poc-early-warning)
        """)

    with tab3:
        st.subheader("📚 Training Dataset Overview")
        df = load_training_data()

        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", f"{df.shape[0]:,}")
            with col2:
                st.metric("Features", df.shape[1])
            with col3:
                if 'hs_diploma' in df.columns:
                    grad_rate = (df['hs_diploma'].sum() / len(df)) * 100
                    st.metric("Graduation Rate", f"{grad_rate:.1f}%")

            with st.expander("📊 View Sample Data"):
                st.dataframe(df.head(100), use_container_width=True)
        else:
            st.info("Training data not available in this deployment.")

    with tab1:
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
                    risk_chart = create_risk_indicator_chart(user_input_df, risk_factors)
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
                        st.write(f"• Attendance Rate: {100 - user_input_dict['pct_days_absent']:.1f}%")
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

                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.exception(e)


if __name__ == "__main__":
    main()
