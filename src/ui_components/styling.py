"""UI styling and reusable components for POC Early Warning System."""
import streamlit as st


def get_custom_css():
    """
    Get custom CSS styles for the application.

    Returns:
        str: CSS styles
    """
    return """
    <style>
        /* Hero Title */
        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        .hero-subtitle {
            font-size: 1.4rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }

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
            margin: 10px 0;
        }

        .metric-card h2 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }

        .metric-card p {
            margin: 5px 0 0 0;
            font-size: 1rem;
            opacity: 0.9;
        }

        /* Info boxes */
        .info-box {
            background-color: #e3f2fd;
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

        /* Danger box */
        .danger-box {
            background-color: #f8d7da;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #dc3545;
            margin: 20px 0;
        }

        /* Navigation card */
        .nav-card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
            border: 2px solid transparent;
        }

        .nav-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
            border-color: #1f77b4;
        }

        .nav-card-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }

        .nav-card-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }

        .nav-card-description {
            font-size: 1rem;
            color: #666;
        }

        /* Section headers */
        .section-header {
            font-size: 2rem;
            font-weight: 600;
            color: #333;
            margin-top: 30px;
            margin-bottom: 20px;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
        }

        /* Metric badge */
        .metric-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            margin: 5px;
        }

        .metric-badge-green {
            background-color: #28a745;
            color: white;
        }

        .metric-badge-yellow {
            background-color: #ffc107;
            color: #333;
        }

        .metric-badge-red {
            background-color: #dc3545;
            color: white;
        }

        /* Table styling */
        .dataframe {
            font-size: 0.9rem;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }

        /* Button styling */
        .stButton>button {
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

        /* Download button */
        .stDownloadButton>button {
            background-color: #28a745;
            color: white;
        }

        .stDownloadButton>button:hover {
            background-color: #218838;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """


def apply_custom_css():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def create_metric_card(label, value, delta=None, delta_color="normal"):
    """
    Create a styled metric card.

    Args:
        label (str): Metric label
        value (str): Metric value
        delta (str, optional): Delta value
        delta_color (str): Color for delta ("normal", "inverse", "off")
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def create_hero_section(title, subtitle, metrics_dict):
    """
    Create hero section with title, subtitle, and key metrics.

    Args:
        title (str): Main title
        subtitle (str): Subtitle text
        metrics_dict (dict): Dictionary of metrics {label: (value, delta)}
    """
    st.markdown(f'<h1 class="hero-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="hero-subtitle">{subtitle}</p>', unsafe_allow_html=True)

    if metrics_dict:
        cols = st.columns(len(metrics_dict))
        for idx, (label, data) in enumerate(metrics_dict.items()):
            with cols[idx]:
                if isinstance(data, tuple):
                    value, delta = data
                    create_metric_card(label, value, delta)
                else:
                    create_metric_card(label, data)


def create_navigation_card(title, description, icon, page_link=None):
    """
    Create a navigation card that links to another page.

    Args:
        title (str): Card title
        description (str): Card description
        icon (str): Emoji icon
        page_link (str, optional): Page to link to
    """
    st.markdown(f"""
    <div class="nav-card">
        <div class="nav-card-icon">{icon}</div>
        <div class="nav-card-title">{title}</div>
        <div class="nav-card-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def create_section_header(text):
    """
    Create a styled section header.

    Args:
        text (str): Header text
    """
    st.markdown(f'<h2 class="section-header">{text}</h2>', unsafe_allow_html=True)


def create_info_box(content, box_type="info"):
    """
    Create a styled info box.

    Args:
        content (str): Box content (HTML allowed)
        box_type (str): Type of box ("info", "success", "warning", "danger")
    """
    box_class = f"{box_type}-box"
    st.markdown(f'<div class="{box_class}">{content}</div>', unsafe_allow_html=True)


def create_metric_badge(text, badge_type="green"):
    """
    Create a colored metric badge.

    Args:
        text (str): Badge text
        badge_type (str): Badge color ("green", "yellow", "red")

    Returns:
        str: HTML for badge
    """
    return f'<span class="metric-badge metric-badge-{badge_type}">{text}</span>'


def display_key_metrics(metrics_data, columns=4):
    """
    Display key metrics in a grid layout.

    Args:
        metrics_data (list): List of tuples (label, value, delta)
        columns (int): Number of columns
    """
    cols = st.columns(columns)

    for idx, data in enumerate(metrics_data):
        col_idx = idx % columns
        with cols[col_idx]:
            if len(data) == 3:
                label, value, delta = data
                st.metric(label, value, delta)
            else:
                label, value = data
                st.metric(label, value)
