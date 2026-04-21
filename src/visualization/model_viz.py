"""Model performance visualization utilities."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def create_confusion_matrix(y_test, y_pred):
    """
    Create interactive confusion matrix visualization.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        plotly.Figure: Interactive confusion matrix heatmap
    """
    cm = confusion_matrix(y_test, y_pred)

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create text annotations
    text = [[f"{cm[i,j]}<br>({cm_percent[i,j]:.1f}%)"
             for j in range(len(cm))]
            for i in range(len(cm))]

    # Create heatmap
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=['Predicted<br>No Diploma', 'Predicted<br>Diploma'],
        y=['Actual<br>No Diploma', 'Actual<br>Diploma'],
        annotation_text=text,
        colorscale='Blues',
        showscale=True,
        hoverinfo='z'
    )

    fig.update_layout(
        title={
            'text': 'Confusion Matrix',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500,
        font=dict(size=12),
        xaxis=dict(side='bottom')
    )

    # Update annotations font
    for annotation in fig.layout.annotations:
        annotation.font.size = 14

    return fig


def create_roc_curve(y_test, y_proba):
    """
    Create ROC curve with AUC score.

    Args:
        y_test: True labels
        y_proba: Predicted probabilities

    Returns:
        plotly.Figure: ROC curve visualization
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<b>FPR</b>: %{x:.3f}<br><b>TPR</b>: %{y:.3f}<extra></extra>'
    ))

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip'
    ))

    fig.update_layout(
        title={
            'text': f'ROC Curve (AUC = {roc_auc:.3f})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        font=dict(size=12)
    )

    return fig


def create_precision_recall_curve(y_test, y_proba):
    """
    Create Precision-Recall curve.

    Args:
        y_test: True labels
        y_proba: Predicted probabilities

    Returns:
        plotly.Figure: Precision-Recall curve visualization
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    fig = go.Figure()

    # PR curve
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name='PR Curve',
        line=dict(color='#2ca02c', width=3),
        hovertemplate='<b>Recall</b>: %{x:.3f}<br><b>Precision</b>: %{y:.3f}<extra></extra>'
    ))

    # Mark threshold = 0.5
    threshold_idx = np.argmin(np.abs(thresholds - 0.5))
    fig.add_trace(go.Scatter(
        x=[recall[threshold_idx]],
        y=[precision[threshold_idx]],
        mode='markers',
        name='Decision Threshold (0.5)',
        marker=dict(color='red', size=12, symbol='star'),
        hovertemplate='<b>Threshold 0.5</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Precision-Recall Curve',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=500,
        showlegend=True,
        hovermode='closest',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        font=dict(size=12)
    )

    return fig


def create_classification_metrics_table(y_test, y_pred):
    """
    Create a DataFrame with classification metrics.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        pd.DataFrame: Classification metrics
    """
    metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        ]
    }

    df = pd.DataFrame(metrics)
    df['Score'] = df['Score'].apply(lambda x: f"{x:.4f}")
    df['Percentage'] = df['Score'].apply(lambda x: f"{float(x)*100:.2f}%")

    return df
