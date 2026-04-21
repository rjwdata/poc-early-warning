# 🎓 POC Early Warning System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/UV-Package%20Manager-DE5D43.svg)](https://github.com/astral-sh/uv)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI-Powered Student Graduation Risk Prediction**

A comprehensive machine learning system that predicts high school graduation outcomes with **90.6% accuracy**, providing actionable insights for early intervention and student support through a professional multi-page web application.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Application Structure](#application-structure)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

---

## 🎯 Overview

### The Problem

Traditional early warning systems use a **one-size-fits-all approach** with generic indicators (attendance <90%, D/F counts, suspensions) that lack specificity about **what students are at risk of**.

### Our AI Solution

This system leverages **machine learning** to create **individualized risk predictions** with:

- 📊 **15 predictive features** across demographics, academics, attendance, and test scores
- 🎯 **90.6% accuracy** in graduation prediction
- 🤖 **AI pattern learning** discovering that ACT scores account for 56% of predictive power
- 📈 **Comprehensive fairness analysis** across demographic subgroups
- 🔄 **Real-time risk assessment** with interactive visualizations

---

## ✨ Key Features

### 🏠 **5-Page Professional Application**

**Executive Summary** - Business-focused dashboard
- Key performance metrics (90.6% accuracy, 42K students, 7 models)
- Model comparison visualizations
- Top predictive features analysis
- Business value proposition

**Technical Details** - Deep technical documentation
- Model architecture & hyperparameters
- Confusion matrix, ROC curves, precision-recall analysis
- Feature importance analysis (top 15 features)
- Comprehensive model comparison (7 algorithms)

**Make Predictions** - Interactive risk assessment
- Organized input forms with 15 student indicators
- Real-time graduation probability predictions
- Risk factor breakdown (6 categories)
- Personalized intervention recommendations
- Downloadable prediction reports (JSON)

**Model Cards** - Transparency & documentation
- Comprehensive fairness analysis by demographic subgroups
- Usage guidelines and limitations
- Bias mitigation strategies
- Intended use and ethical considerations

**Data Quality** - Monitoring & reports
- Embedded Evidently AI quality reports
- Data drift detection
- Automated quality tests (92% pass rate)
- Dataset validation metrics

### 🤖 **Machine Learning Pipeline**
- XGBoost classifier with Bayesian hyperparameter optimization
- Automated evaluation of 7 algorithms
- Feature importance analysis and interpretability
- Modular pipeline: Data Ingestion → Transformation → Training

### 🔍 **AI Capabilities**
- Pattern discovery exceeding human-designed rules
- Individualized probability scores (not binary thresholds)
- Automated feature learning and importance ranking
- Continuous improvement through retraining

---

## 🚀 Installation

### Prerequisites

- Python 3.8+ 
- Git
- [UV](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Install

```bash
# Clone repository
git clone https://github.com/rjwdata/poc-early-warning.git
cd poc-early-warning

# Install with UV (recommended - 10-100x faster)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .

# OR install with Make
make install

# OR traditional pip
pip install -e .
```

---

## ⚡ Quick Start

### Run the Application

```bash
# Using Make (recommended)
make run-app

# OR directly
streamlit run app_main.py
```

The app opens at **http://localhost:8501** with 5 pages:

- 🏠 **Executive Summary** - Start here for business overview
- 🔬 **Technical Details** - Model architecture and performance
- 🎯 **Make Predictions** - Interactive student risk assessment  
- 📋 **Model Cards** - Comprehensive documentation
- 📊 **Data Quality** - Monitoring reports

### Train the Model

```bash
make train
# OR
python src/components/data_ingestion_eda.py
```

### Available Commands

```bash
make help          # Show all commands
make install-dev   # Install with dev dependencies
make test          # Run test suite
make lint          # Run linters
make format        # Format code
make clean         # Remove build artifacts
```

---

## 🗂️ Application Structure

### Navigation

The unified multi-page app is organized into three sections:

**🏠 Overview**
- Executive Summary (default landing page)

**Technical Documentation**
- Technical Details
- Model Cards  
- Data Quality

**Applications**
- Make Predictions

### Page Details

| Page | Purpose | Key Content |
|------|---------|-------------|
| **Executive Summary** | Business stakeholders | KPIs, model comparison, top features, ROI |
| **Technical Details** | Data scientists | Architecture, confusion matrix, ROC curves, feature importance |
| **Make Predictions** | End users | Interactive forms, real-time predictions, risk analysis |
| **Model Cards** | Compliance/ethics | Fairness analysis, limitations, usage guidelines |
| **Data Quality** | ML engineers | Evidently reports, drift detection, quality metrics |

---

## 📊 Model Performance

### Best Model: XGBoost Classifier

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 90.6% | Correctly predicts 9 out of 10 students |
| **Precision** | 94.7% | Low false positives |
| **Recall** | 93.9% | Catches most at-risk students |
| **F1-Score** | 94.3% | Balanced performance |

### Model Comparison

| Model | Accuracy | Precision | Recall | Training Time |
|-------|----------|-----------|--------|---------------|
| **XGBoost** ⭐ | **90.6%** | **94.7%** | **93.9%** | 45.2s |
| Random Forest | 90.5% | 95.2% | 93.3% | 38.7s |
| KNN | 87.3% | 92.7% | 91.7% | 2.1s |
| Logistic Regression | 85.6% | 96.4% | 87.2% | 5.4s |
| Baseline | 81.3% | 100.0% | 81.3% | <0.1s |

### Dataset

- **Training**: 42,353 students (80%)
- **Testing**: 10,589 students (20%)
- **Features**: 15 predictive indicators
- **Target**: High school diploma (binary)
- **Class Distribution**: 81.2% graduated, 18.8% at-risk

### Top 5 Predictive Features

1. **ACT English Score** (30.4%) - Strongest single predictor
2. **ACT Composite Score** (15.5%) - Overall test performance
3. **GPA** (9.2%) - Academic achievement
4. **ACT Reading** (7.8%) - Reading proficiency
5. **ACT Math** (6.5%) - Math proficiency

**Insight**: Standardized tests account for **56.3%** of total predictive power.

### Fairness Analysis

Performance across demographic subgroups:

| Subgroup | Accuracy | Max Disparity |
|----------|----------|---------------|
| Overall | 90.6% | - |
| Gender | 89.8% - 91.4% | 1.6% |
| Race/Ethnicity | 89.5% - 91.8% | 2.3% |
| Socioeconomic (FRPL) | 88.9% - 92.1% | **3.2%** |

All subgroups maintain >88.9% accuracy with transparent disparity tracking.

---

## 📁 Project Structure

```
poc-early-warning/
├── app_main.py                    # Main application entry point
├── pages/                         # Multi-page application
│   ├── 01_executive_summary.py   # Business dashboard
│   ├── 02_technical_details.py   # Technical deep dive
│   ├── 03_predictions.py         # Interactive predictions
│   ├── 04_model_cards.py         # Documentation
│   └── 05_data_quality.py        # Quality reports
├── src/
│   ├── components/               # ML pipeline
│   │   ├── data_ingestion_eda.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py   # Inference API
│   ├── visualization/            # Plotly charts
│   │   ├── model_viz.py          # Confusion matrix, ROC curves
│   │   ├── feature_viz.py        # Feature importance
│   │   ├── comparison_viz.py     # Model comparison
│   │   └── fairness_viz.py       # Subgroup analysis
│   ├── ui_components/
│   │   └── styling.py            # Shared CSS & components
│   └── utils.py                  # Utility functions
├── artifacts/                    # Trained models & reports
│   ├── model.pkl                 # XGBoost model (327 KB)
│   ├── preprocessor.pkl          # Preprocessing pipeline
│   ├── model_performance.json    # Model comparison data
│   ├── test_predictions.json     # Cached predictions
│   └── *.html                    # Evidently reports
├── config/
│   └── params.yaml               # Model configuration
├── data/                         # Training & test data
├── notebooks/                    # Jupyter notebooks
├── scripts/
│   └── prepare_artifacts.py      # Generate artifacts
├── Makefile                      # Development commands
├── pyproject.toml                # Python configuration
└── README.md                     # This file
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Install with dev dependencies
make install-dev

# Setup pre-commit hooks
make setup-hooks

# Run tests
make test

# Format code
make format

# Run linters
make lint
```

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes

# 3. Format and test
make format
make test

# 4. Commit and push
git add .
git commit -m "Add my feature"
git push origin feature/my-feature
```

### Key Design Patterns

- **Factory Pattern**: Safe model instantiation without `eval()`
- **Pipeline Pattern**: Modular data processing
- **Configuration Pattern**: YAML-based centralized config
- **Caching**: `@st.cache_resource` and `@st.cache_data` for performance

### Code Quality Tools

- **Black**: Code formatting (100 char line length)
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Pytest**: Testing framework

---

## 🤖 AI/ML Techniques

This POC demonstrates:

- **Supervised Learning**: XGBoost gradient boosted decision trees
- **Automated Feature Learning**: Model discovers optimal feature combinations
- **Bayesian Optimization**: Hyperparameter tuning with Hyperopt (100 iterations)
- **Ensemble Methods**: Evaluates 7 algorithms, selects best performer
- **Fairness Analysis**: Bias detection across demographic subgroups
- **Explainability**: Feature importance rankings for interpretability

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Data Source**: Strategic Data Project, Harvard University CEPR

**Technologies**:
- [XGBoost](https://xgboost.readthedocs.io/) - ML framework
- [Streamlit](https://streamlit.io) - Web application
- [Evidently AI](https://evidentlyai.com) - ML monitoring
- [Plotly](https://plotly.com) - Interactive visualizations
- [UV](https://github.com/astral-sh/uv) - Package manager

---

## 📧 Contact

**Maintainer**: hawkeye  
**GitHub**: [@rjwdata](https://github.com/rjwdata)

**Support**:
- 🐛 [Issue Tracker](https://github.com/rjwdata/poc-early-warning/issues)
- 💬 [Discussions](https://github.com/rjwdata/poc-early-warning/discussions)

---

<div align="center">

**Made with ❤️ by hawkeye**

⭐ If you found this project helpful, please star the repository!

[⬆ Back to Top](#-poc-early-warning-system)

</div>
