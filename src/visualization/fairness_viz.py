"""Fairness and bias analysis visualization utilities."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_subgroup_performance(y_test, y_pred, demographic_df, subgroups):
    """
    Calculate performance metrics for different demographic subgroups.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        demographic_df: DataFrame with demographic features
        subgroups: Dict mapping subgroup name to (column, value) tuples

    Returns:
        pd.DataFrame: Subgroup performance metrics
    """
    results = []

    for subgroup_name, (column, value) in subgroups.items():
        if column is None:
            # Overall performance
            mask = np.ones(len(y_test), dtype=bool)
        else:
            # Specific subgroup
            mask = demographic_df[column] == value

        y_test_sub = y_test[mask]
        y_pred_sub = y_pred[mask]

        if len(y_test_sub) > 0:
            results.append({
                'Subgroup': subgroup_name,
                'N': len(y_test_sub),
                'Accuracy': accuracy_score(y_test_sub, y_pred_sub),
                'Precision': precision_score(y_test_sub, y_pred_sub, zero_division=0),
                'Recall': recall_score(y_test_sub, y_pred_sub, zero_division=0),
                'F1-Score': f1_score(y_test_sub, y_pred_sub, zero_division=0)
            })

    df = pd.DataFrame(results)

    # Format percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        df[f'{col}_pct'] = df[col].apply(lambda x: f"{x*100:.1f}%")

    return df


def create_subgroup_comparison_chart(subgroup_df, metric='Accuracy'):
    """
    Create grouped bar chart comparing subgroup performance.

    Args:
        subgroup_df: DataFrame with subgroup metrics
        metric: Metric to visualize

    Returns:
        plotly.Figure: Subgroup comparison chart
    """
    # Sort by metric
    df_sorted = subgroup_df.sort_values(metric, ascending=True)

    # Color code: green for high, yellow for medium, red for low
    def get_color(value):
        if value >= 0.90:
            return '#28a745'  # Green
        elif value >= 0.85:
            return '#ffc107'  # Yellow
        else:
            return '#dc3545'  # Red

    colors = [get_color(v) for v in df_sorted[metric]]

    fig = go.Figure(go.Bar(
        x=df_sorted[metric],
        y=df_sorted['Subgroup'],
        orientation='h',
        marker=dict(color=colors),
        text=df_sorted[f'{metric}_pct'],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>' + metric + ': %{x:.4f}<br>N: ' +
                     df_sorted['N'].astype(str) + '<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': f'{metric} by Demographic Subgroup',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title=metric,
        yaxis_title='',
        height=max(400, len(df_sorted) * 40),
        margin=dict(l=150, r=50, t=80, b=50),
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0.02)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', range=[0.7, 1.0])
    )

    return fig


def create_disparity_metrics_table(subgroup_df):
    """
    Calculate disparity metrics between subgroups.

    Args:
        subgroup_df: DataFrame with subgroup metrics

    Returns:
        pd.DataFrame: Disparity analysis
    """
    overall_row = subgroup_df[subgroup_df['Subgroup'] == 'Overall']

    if len(overall_row) == 0:
        return pd.DataFrame()

    overall_metrics = {
        'Accuracy': overall_row['Accuracy'].values[0],
        'Precision': overall_row['Precision'].values[0],
        'Recall': overall_row['Recall'].values[0],
        'F1-Score': overall_row['F1-Score'].values[0]
    }

    # Calculate disparities
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    disparities = []

    for metric in metrics:
        values = subgroup_df[subgroup_df['Subgroup'] != 'Overall'][metric]
        if len(values) > 0:
            max_val = values.max()
            min_val = values.min()
            gap = max_val - min_val
            max_group = subgroup_df[subgroup_df[metric] == max_val]['Subgroup'].values[0]
            min_group = subgroup_df[subgroup_df[metric] == min_val]['Subgroup'].values[0]

            disparities.append({
                'Metric': metric,
                'Overall': f"{overall_metrics[metric]*100:.1f}%",
                'Max': f"{max_val*100:.1f}%",
                'Max Group': max_group,
                'Min': f"{min_val*100:.1f}%",
                'Min Group': min_group,
                'Gap': f"{gap*100:.1f}%"
            })

    return pd.DataFrame(disparities)


def create_fairness_heatmap(subgroup_df):
    """
    Create heatmap showing all metrics across subgroups.

    Args:
        subgroup_df: DataFrame with subgroup metrics

    Returns:
        plotly.Figure: Fairness heatmap
    """
    # Prepare data
    subgroups = subgroup_df['Subgroup'].values
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # Create matrix
    z_data = []
    for metric in metrics:
        z_data.append(subgroup_df[metric].values)

    # Create annotations
    annotations = []
    for i, metric in enumerate(metrics):
        for j, subgroup in enumerate(subgroups):
            value = subgroup_df[subgroup_df['Subgroup'] == subgroup][metric].values[0]
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{value*100:.1f}%",
                    showarrow=False,
                    font=dict(color='white' if value < 0.85 else 'black', size=11)
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=subgroups,
        y=metrics,
        colorscale='RdYlGn',
        zmid=0.85,
        zmin=0.70,
        zmax=1.0,
        colorbar=dict(title='Score'),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Score: %{z:.4f}<extra></extra>'
    ))

    # Add annotations
    for annotation in annotations:
        fig.add_annotation(annotation)

    fig.update_layout(
        title={
            'text': 'Model Performance Across Demographics',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=400,
        font=dict(size=12),
        xaxis=dict(side='bottom')
    )

    return fig


def create_subgroup_sample_size_chart(subgroup_df):
    """
    Create bar chart showing sample sizes for each subgroup.

    Args:
        subgroup_df: DataFrame with subgroup metrics

    Returns:
        plotly.Figure: Sample size chart
    """
    df_sorted = subgroup_df.sort_values('N', ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted['N'],
        y=df_sorted['Subgroup'],
        orientation='h',
        marker=dict(color='#1f77b4'),
        text=df_sorted['N'],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Sample Size: %{x:,}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Sample Size by Subgroup',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Number of Students',
        yaxis_title='',
        height=max(400, len(df_sorted) * 40),
        margin=dict(l=150, r=50, t=80, b=50),
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0.02)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )

    return fig
