# AutoML Pipeline Orchestrator

> From raw CSV to deployed FastAPI endpoint in one command. No Jupyter notebooks, no copy-paste.

## 🚨 The Pain We Solve

A data scientist at an e-commerce company spent 3 weeks on:
1. Manual feature engineering (copy-paste from Kaggle)
2. GridSearchCV with 4 models
3. Writing Flask boilerplate
4. Dockerizing... then discovering model doesn't generalize

This agent automates the entire pipeline with **automatic feature engineering**, **Bayesian optimization**, and **deployment-ready artifacts**.

## 🏗️ Pipeline Architecture

```
Raw CSV
  ↓
Data Profiler Agent → detects types, skew, leakage, drift
  ↓
Feature Engineer Agent → generates 150+ features, selects top-K via SHAP
  ↓
Model Selector Agent → tests 15 algorithms with stratified CV
  ↓
Hyperopt Agent → Bayesian optimization with early stopping
  ↓
Deploy Agent → FastAPI + OpenAPI spec + Dockerfile + monitoring
```

## 🔧 What Makes It Different

- **Leakage Detection**: Automatically flags target leakage (e.g., future timestamps in features).
- **Auto-Feature Cross**: Creates polynomial + interaction features, then selects via mutual information.
- **Model Stacking**: Auto-builds ensemble of top-3 models with learned weights.
- **Explainability**: SHAP summary plots + feature importance report generated automatically.
- **A/B Test Ready**: Deploys shadow endpoint for safe rollout.

## 📊 Token Consumption

| Dataset | Rows | Tokens | Time |
|---|---|---|---|
| Small (<10K) | ~50 cols | 80K | 4 min |
| Medium (~500K) | ~200 cols | 650K | 22 min |
| Large (~5M) | ~500 cols | 2.8M | 1.2 hrs |
| **Monthly** | — | **~15M** | — |

## 📈 Results

E-commerce churn prediction:
- Baseline (manual): **87.2% AUC**, 2 weeks dev time
- AutoML agent: **94.1% AUC**, **6 hours** end-to-end
- Auto-generated **47 interaction features**
- Deployed to production with **zero handoff friction**

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python automl.py --data ./churn.csv --target is_churn --time-limit 1800 --deploy
```

## 🛠️ Tech Stack

- Python 3.11 + MiMo API (planning + code gen)
- Optuna, XGBoost, LightGBM, scikit-learn
- MLflow tracking
- FastAPI + Docker
