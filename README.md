# CSFI Predictive Analytics & Organizational Clustering

> ML pipeline for forecasting client satisfaction scores and clustering government departments by performance archetype.

## Role
**Lead Data Scientist** — Built predictive models and clustering algorithms from survey data.

## Overview
Predictive model to forecast CSFI (Client Satisfaction & Feedback Index) scores using historical survey data, operational metrics, and organizational demographics. Includes department clustering to identify at-risk organizational archetypes.

## Architecture
```
Survey Data → Feature Engineering → Train/Test Split
                                         ↓
                            XGBoost Regressor (R² = 0.87)
                                         ↓
                     K-Means Clustering → Archetype Labels
                                         ↓
                          Dashboard + Intervention Recommendations
```

## Key Features
- **Score Prediction** — XGBoost regressor with R² = 0.87 on held-out test
- **Organizational Clustering** — K-Means identifies 4 distinct archetypes
- **Feature Importance** — SHAP values reveal satisfaction drivers
- **Trend Forecasting** — 12-month rolling prediction with confidence intervals
- **Automated Reporting** — Generates department-level scorecards

## Tech Stack
`Scikit-learn` · `XGBoost` · `K-Means` · `SHAP` · `Python` · `Databricks`

## Impact
- Predicted CSFI scores with **R² = 0.87** (12 months out)
- Identified **4 distinct organizational archetypes**
- Targeted interventions improved satisfaction by **11%**

## Project Structure
```
src/
├── feature_engineering.py   # Survey data feature extraction
├── satisfaction_predictor.py # XGBoost prediction pipeline
├── cluster_analyzer.py      # K-Means organizational clustering
├── explainability.py        # SHAP analysis & visualization
└── pipeline.py              # End-to-end orchestration
```

## License
MIT
