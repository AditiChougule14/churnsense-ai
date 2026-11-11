
# Model Card — ChurnSense AI

> **Predict Churn. Understand Why. Drive Retention.**

## Model Details

- **Project Name:** ChurnSense AI
- **Type:** scikit-learn `Pipeline` (feature engineering → impute/scale + one-hot → classifier).
- **Default Estimator:** `LogisticRegression` (with Random Forest candidate evaluated).
- **Version:** 2.0 · Serialized with scikit-learn 1.8.0 (`models/churn_pipeline.pkl`).
- **Selection:** Best of Logistic Regression vs. Random Forest by 5-fold cross-validated ROC-AUC on the training split.
- **Explainability:** SHAP (`LinearExplainer` and `TreeExplainer`) for individual and global explanations.
- **Owner / Author:** Aditi Chougule
- **Repository:** [churnsense-ai](https://github.com/aditichougule/churnsense-ai)

## Intended Use

- **Primary Use:** Estimate the probability that a customer will churn, prioritizing proactive **retention outreach**.
- **Intended Users:** Retention teams, CRM managers, customer success specialists, and business analysts.
- **Out of Scope:** Automated punitive actions or pricing discrimination without human review; domains outside customer churn; decisions regarding legally protected attributes.

## Training Data

- **Dataset:** Telco Customer Churn (IBM sample dataset via Kaggle).
- **Dataset Size:** 7,043 customer records; churn rate ≈ 26.5% (imbalanced, handled via `class_weight="balanced"`).
- **Splitting:** 80/20 stratified split; held-out test set evaluated once for final reporting.
- **Features (19):** Demographic, service subscription, account, and billing attributes.

## Evaluation Metrics (Held-Out Test Set, n = 1,409)

| Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|
| 0.731 | 0.496 | **0.797** | 0.611 | **0.847** | 0.851 |

- **Decision Threshold:** 0.5 (default). Can be tuned based on customer acquisition vs. retention cost ratios.
- **Live Metrics:** Persisted in `models/metadata.json` and `reports/model_comparison.csv`.

## Limitations

- Trained on a point-in-time snapshot; does not incorporate continuous drift monitoring or temporal validation out of the box.
- Precision ≈ 0.50: roughly half of flagged customers might not churn without intervention — acceptable for supportive outreach, not for adverse actions.
- Generalizability to non-telecom domains requires retraining via the custom CSV AutoML module.

## Ethical Considerations & Explainability

- Attributes such as `gender` and `SeniorCitizen` are included; predictions should be audited for disparate impact before real-world deployment.
- **Transparency via SHAP:** Every prediction is explainable at the customer level (`src/explain.py`), allowing stakeholders to understand the factors driving churn risk.

## Maintenance & Retraining

- **Retraining:** Run `python train_model.py` to retrain, validate, and serialize pipeline artifacts.
- **Automated Quality Gate:** Test suite enforces minimum performance criteria (ROC-AUC ≥ 0.78 and recall ≥ 0.45).
