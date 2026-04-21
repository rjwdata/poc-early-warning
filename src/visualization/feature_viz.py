"""Feature importance visualization utilities."""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os


def get_feature_importance_data():
    """
    Extract feature importance from trained XGBoost model.

    Returns:
        pd.DataFrame: Feature importance data with columns ['feature', 'importance']
    """
    try:
        model = joblib.load('artifacts/model.pkl')
        preprocessor = joblib.load('artifacts/preprocessor.pkl')

        # Get feature importances
        importances = model.feature_importances_

        # Get feature names from preprocessor
        feature_names = preprocessor.get_feature_names_out()

        # Create DataFrame and sort
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return df

    except FileNotFoundError as e:
        # Return dummy data if files not found
        print(f"Warning: Could not load model files: {e}")
        return pd.DataFrame({
            'feature': ['scale_score_11_eng', 'scale_score_11_comp', 'gpa',
                       'scale_score_11_read', 'scale_score_11_math'],
            'importance': [0.304, 0.155, 0.092, 0.078, 0.065]
        })


def create_feature_importance_chart(top_n=15):
    """
    Create horizontal bar chart of feature importances.

    Args:
        top_n (int): Number of top features to display

    Returns:
        plotly.Figure: Feature importance bar chart
    """
    df = get_feature_importance_data().head(top_n)

    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker=dict(
            color=df['importance'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(
                title='Importance',
                titleside='right',
                tickmode='linear',
                tick0=0,
                dtick=0.05
            )
        ),
        text=df['importance'].apply(lambda x: f'{x:.3f}'),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': f'Top {top_n} Most Important Features',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Importance Score',
        yaxis_title='',
        height=max(400, top_n * 30),  # Dynamic height based on number of features
        margin=dict(l=250, r=50, t=80, b=50),  # Extra left margin for long feature names
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0.02)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )

    return fig


def create_feature_categories_pie():
    """
    Create pie chart showing feature importance by category.

    Returns:
        plotly.Figure: Pie chart of feature categories
    """
    df = get_feature_importance_data()

    # Categorize features
    def categorize_feature(feature_name):
        if 'scale_score' in feature_name or 'act' in feature_name.lower():
            return 'Standardized Tests'
        elif 'gpa' in feature_name.lower():
            return 'Academic Performance'
        elif 'math_ss' in feature_name or 'read_ss' in feature_name:
            return 'Academic Performance'
        elif 'iep' in feature_name or 'ell' in feature_name or 'alternative' in feature_name:
            return 'Support Services'
        elif 'race' in feature_name or 'male' in feature_name or 'frpl' in feature_name:
            return 'Demographics'
        elif 'absent' in feature_name or 'attendance' in feature_name:
            return 'Attendance'
        elif 'ap' in feature_name.lower():
            return 'Academic Performance'
        else:
            return 'Other'

    df['category'] = df['feature'].apply(categorize_feature)

    # Sum importances by category
    category_importance = df.groupby('category')['importance'].sum().reset_index()
    category_importance = category_importance.sort_values('importance', ascending=False)

    # Calculate percentages
    total_importance = category_importance['importance'].sum()
    category_importance['percentage'] = (category_importance['importance'] / total_importance * 100)

    fig = go.Figure(data=[go.Pie(
        labels=category_importance['category'],
        values=category_importance['importance'],
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set2),
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Importance: %{value:.3f}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title={
            'text': 'Feature Importance by Category',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=500,
        showlegend=True,
        font=dict(size=12)
    )

    return fig


def get_feature_category_summary():
    """
    Get summary statistics of feature importance by category.

    Returns:
        pd.DataFrame: Category summary with percentages
    """
    df = get_feature_importance_data()

    # Categorize features
    def categorize_feature(feature_name):
        if 'scale_score' in feature_name or 'act' in feature_name.lower():
            return 'Standardized Tests'
        elif 'gpa' in feature_name.lower() or 'math_ss' in feature_name or 'read_ss' in feature_name or 'ap' in feature_name.lower():
            return 'Academic Performance'
        elif 'iep' in feature_name or 'ell' in feature_name or 'alternative' in feature_name:
            return 'Support Services'
        elif 'race' in feature_name or 'male' in feature_name or 'frpl' in feature_name:
            return 'Demographics'
        elif 'absent' in feature_name or 'attendance' in feature_name:
            return 'Attendance'
        else:
            return 'Other'

    df['category'] = df['feature'].apply(categorize_feature)

    # Sum importances by category
    category_importance = df.groupby('category')['importance'].sum().reset_index()
    category_importance = category_importance.sort_values('importance', ascending=False)

    # Calculate percentages
    total_importance = category_importance['importance'].sum()
    category_importance['percentage'] = (category_importance['importance'] / total_importance * 100)
    category_importance['percentage_str'] = category_importance['percentage'].apply(lambda x: f"{x:.1f}%")

    return category_importance
