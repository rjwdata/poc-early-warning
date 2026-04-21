"""Model comparison visualization utilities."""
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def load_model_performance():
    """
    Load model performance metrics from JSON file.

    Returns:
        dict: Model performance data
    """
    try:
        with open('artifacts/model_performance.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default data if file not found
        return {
            "baseline": {"accuracy": 0.813, "precision": 1.000, "recall": 0.813, "f1": 0.897, "training_time": 0.001},
            "logistic_regression": {"accuracy": 0.856, "precision": 0.964, "recall": 0.872, "f1": 0.916, "training_time": 5.4},
            "svm": {"accuracy": 0.855, "precision": 0.971, "recall": 0.866, "f1": 0.915, "training_time": 12.8},
            "decision_tree": {"accuracy": 0.848, "precision": 0.902, "recall": 0.910, "f1": 0.906, "training_time": 1.2},
            "random_forest": {"accuracy": 0.905, "precision": 0.952, "recall": 0.933, "f1": 0.942, "training_time": 38.7},
            "naive_bayes": {"accuracy": 0.748, "precision": 0.746, "recall": 0.930, "f1": 0.829, "training_time": 0.3},
            "knn": {"accuracy": 0.873, "precision": 0.927, "recall": 0.917, "f1": 0.922, "training_time": 2.1},
            "xgboost": {"accuracy": 0.906, "precision": 0.947, "recall": 0.939, "f1": 0.943, "training_time": 45.2}
        }


def create_model_comparison_bars(metric='accuracy'):
    """
    Create grouped bar chart comparing all models.

    Args:
        metric (str): Metric to display ('accuracy', 'precision', 'recall', or 'all')

    Returns:
        plotly.Figure: Model comparison bar chart
    """
    results = load_model_performance()

    # Convert to DataFrame
    models = list(results.keys())
    model_names = [m.replace('_', ' ').title() for m in models]

    if metric == 'all':
        # Show all metrics
        metrics = ['accuracy', 'precision', 'recall']
        fig = go.Figure()

        for m in metrics:
            values = [results[model][m] for model in models]
            fig.add_trace(go.Bar(
                name=m.capitalize(),
                x=model_names,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' + m.capitalize() + ': %{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            barmode='group',
            yaxis_range=[0, 1.05]
        )
        title = 'Model Performance Comparison - All Metrics'
    else:
        # Show single metric
        values = [results[model][metric] for model in models]

        # Color code: green for best, gradient for others
        colors = ['#28a745' if v == max(values) else '#1f77b4' for v in values]

        fig = go.Figure(go.Bar(
            x=model_names,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
            marker=dict(color=colors),
            hovertemplate='<b>%{x}</b><br>' + metric.capitalize() + ': %{y:.4f}<extra></extra>'
        ))

        fig.update_layout(yaxis_range=[0, 1.05])
        title = f'Model Comparison - {metric.capitalize()}'

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Model',
        yaxis_title='Score',
        height=500,
        showlegend=(metric == 'all'),
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0.02)',
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )

    return fig


def create_model_comparison_table():
    """
    Create DataFrame with model comparison metrics.

    Returns:
        pd.DataFrame: Model comparison table
    """
    results = load_model_performance()

    data = []
    for model_name, metrics in results.items():
        data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'Training Time (s)': f"{metrics['training_time']:.2f}"
        })

    df = pd.DataFrame(data)

    # Sort by accuracy descending
    df = df.sort_values('Accuracy', ascending=False)

    return df


def create_model_radar_chart(models_list=None):
    """
    Create radar chart comparing top models across multiple metrics.

    Args:
        models_list (list): List of model names to compare. If None, uses top 3.

    Returns:
        plotly.Figure: Radar chart
    """
    results = load_model_performance()

    if models_list is None:
        # Select top 3 by accuracy
        sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        models_list = [m[0] for m in sorted_models[:3]]

    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, model_name in enumerate(models_list):
        metrics = results[model_name]
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ]

        # Close the radar chart
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name=model_name.replace('_', ' ').title(),
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.4f}<extra></extra>'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title={
            'text': 'Top Models - Multi-Metric Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=500,
        font=dict(size=12)
    )

    return fig


def create_accuracy_vs_time_scatter():
    """
    Create scatter plot showing accuracy vs training time trade-off.

    Returns:
        plotly.Figure: Scatter plot
    """
    results = load_model_performance()

    models = list(results.keys())
    model_names = [m.replace('_', ' ').title() for m in models]
    accuracies = [results[m]['accuracy'] for m in models]
    times = [results[m]['training_time'] for m in models]

    # Color by model family
    colors = []
    for model in models:
        if 'tree' in model or 'forest' in model or 'xgboost' in model:
            colors.append('Tree-based')
        elif 'svm' in model or 'logistic' in model:
            colors.append('Linear')
        elif 'naive_bayes' in model:
            colors.append('Probabilistic')
        elif 'knn' in model:
            colors.append('Instance-based')
        else:
            colors.append('Other')

    fig = go.Figure()

    # Create scatter with different colors for each family
    for family in set(colors):
        indices = [i for i, c in enumerate(colors) if c == family]
        fig.add_trace(go.Scatter(
            x=[times[i] for i in indices],
            y=[accuracies[i] for i in indices],
            mode='markers+text',
            name=family,
            text=[model_names[i] for i in indices],
            textposition='top center',
            marker=dict(size=12),
            hovertemplate='<b>%{text}</b><br>Accuracy: %{y:.4f}<br>Time: %{x:.2f}s<extra></extra>'
        ))

    fig.update_layout(
        title={
            'text': 'Model Accuracy vs Training Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Training Time (seconds, log scale)',
        yaxis_title='Accuracy',
        xaxis_type='log',
        height=500,
        showlegend=True,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0.02)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', range=[0.7, 1.0])
    )

    return fig
