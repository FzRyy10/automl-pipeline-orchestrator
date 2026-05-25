# AutoML Pipeline Orchestrator

> Autonomous machine learning agent that builds, tunes, and deploys ML pipelines end-to-end.

## 🎯 What It Solves

Building production ML pipelines requires expertise in feature engineering, model selection, hyperparameter tuning, and deployment. This agent automates the entire workflow while maintaining interpretability.

## 🏗️ Architecture

```
Dataset → Profiler → [Feature Engineer | Model Selector | Hyperopt Agent]
                                    ↓
                        Pipeline Compiler → Deployer
```

## 🔧 Core Features

- **Data Profiler**: Auto-detects types, distributions, quality issues
- **Feature Engineer**: Generates 100+ features, selects top-K via SHAP
- **Model Selector**: Tests 15+ algorithms with cross-validation
- **Hyperopt Agent**: Bayesian optimization with early stopping
- **Auto-Deployer**: Generates FastAPI service + Docker image

## 📊 Token Consumption

- Small dataset (<10K rows): ~100K tokens
- Medium dataset (~1M rows): ~800K tokens
- Large dataset (~10M rows): ~3M tokens
- **Monthly average**: 10-20M tokens for data science team

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python automl.py --data ./sales.csv --target revenue --time-limit 3600
```

## 📈 Results

Used by an e-commerce analytics team:
- Model development time: 2 weeks → 6 hours
- Achieved 94.3% accuracy (vs 91% manual baseline)
- Auto-generated 47 engineered features

## 🛠️ Tech Stack

- Python 3.11+
- MiMo API (planning + code generation)
- Optuna, scikit-learn, XGBoost
- MLflow for experiment tracking
