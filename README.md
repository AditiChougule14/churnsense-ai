# ChurnSense AI

> **Predict Churn. Understand Why. Drive Retention.**

An end-to-end Explainable AI (XAI) platform for customer churn prediction, featuring dual operational modes: **Demo (Telco Benchmark)** and **AutoML (Custom CSV)**. Powered by Scikit-learn, Streamlit, and SHAP to deliver transparent, actionable retention intelligence.

**Author:** Aditi Chougule  
**Repository:** [churnsense-ai](https://github.com/aditichougule/churnsense-ai)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)
![SHAP](https://img.shields.io/badge/SHAP-XAI-blueviolet.svg)
![Streamlit]![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🛠️ Case Study: From a Broken Model to ROC-AUC 0.85

This project originated with a critical architectural audit and transformation:
The initial baseline was non-functional — trained on synthetic random data, it predicted "No Churn" for every customer, while a train/serve bug silently dropped 5 inputs. An end-to-end refactoring (real Telco data + a single fit-once / serve-once pipeline) resolved the issues:

| Metric (held-out test set) | Baseline | ChurnSense AI |
|---|---|---|
| ROC-AUC | 0.49 (random) | **0.847** |
| Recall (churners caught) | 0.00 | **0.797** |
| F1-score | 0.00 | **0.611** |
| Customers flagged as churn | 0 / 7,043 | Realistic (~80% caught) |

Every prediction in **ChurnSense AI** includes a **per-customer SHAP explanation** (waterfall and contribution analysis):

![Per-customer explanation](screenshots/fig_explanation_example.png)

---

## 📋 Table of Contents
- [Overview & Core Capabilities](#-overview--core-capabilities)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Visuals](#-visuals)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Service](#-api-service)
- [Deployment](#-deployment)
- [Model Performance](#-model-performance)
- [Business Impact](#-business-impact)
- [Explainable AI (SHAP)](#-explainable-ai-shap)
- [Testing](#-testing)
- [Author & License](#-author--license)

---

## 🎯 Overview & Core Capabilities

Customer churn is among the costliest business challenges. Identifying at-risk customers **before** they leave enables targeted, cost-effective retention. **ChurnSense AI** bridges machine learning and business action through full lifecycle modeling and explainability:

- 🎯 **Demo (Telco Mode)**: Pre-trained on the Telco benchmark dataset across 5 dedicated dashboard pages (Overview, Churn Insights, Make Prediction, Model Details, Data Overview).
- 📤 **Custom CSV AutoML Mode**: Upload any CSV dataset with a binary target. Automatically detects numeric and categorical features, handles missing values, detects and ignores ID columns, trains models, evaluates performance, and provides instant interactive predictions.
- ⚖️ **Model Comparison**: Train, evaluate, and compare Logistic Regression and Random Forest architectures using 5-fold cross-validation on the training set to prevent test leakage.
- 🔍 **Explainable AI (SHAP-Based)**:
  - **Global Explanations**: SHAP feature importance bar charts and summary beeswarm visualizations showing overall model drivers across the dataset.
  - **Individual Explanations**: Customer-level SHAP waterfall / contribution plots showing exact positive and negative feature contributions toward churn vs. retention.
  - **Domain Aggregation**: One-hot encoded feature contributions are mapped back to their original domain columns for intuitive business interpretation.
- 💡 **Actionable Insights & Risk Scoring**: Classifies customer risk into Low, Medium, and High tiers with tailored retention strategies and downloadable trained pipelines (`.pkl`).

---

## ✨ Key Features

1. **Dual Operational Modes**:
   - Seamlessly switch between the bundled Telco benchmark demo and custom CSV training in a single unified Streamlit interface.
2. **Robust Preprocessing & Guardrails**:
   - Fit-once / serve-once architecture eliminates train-serve skew.
   - Outlier protection, safe imputation, and `OneHotEncoder(handle_unknown="ignore")`.
   - Data validation guards against synthetic data, constant columns, non-binary targets, and single-class distributions.
3. **Transparent Model Interpretability**:
   - Native SHAP integration supporting both Linear and Tree-based explainers (`LinearExplainer`, `TreeExplainer`).
   - Mathematically consistent SHAP baselines using representative reference datasets.
4. **Interactive Dashboard & REST API**:
   - Clean, modern UI built with Streamlit and Plotly.
   - Production-ready FastAPI microservice with Pydantic schema validation and batch scoring endpoints.

---

## 💻 Technology Stack

- **Python** (3.11+)
- **Streamlit** (interactive web dashboard)
- **Scikit-learn** (pipelines, preprocessing, model training & evaluation)
- **SHAP** (Explainable AI: TreeExplainer, LinearExplainer, waterfall, beeswarm)
- **Plotly** (interactive visualizations & charts)
- **Pandas & NumPy** (data manipulation & numerical computing)
- **Matplotlib** (static figures & report visualizations)
- **FastAPI & Pydantic** (REST API service & schema validation)

---

## 📊 Dataset

[Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) (IBM sample data):
- **7,043** customer records, **19** input features + binary `Churn` target
- **Churn rate ≈ 26.5%** (imbalanced) — handled via `class_weight="balanced"`
- Blank `TotalCharges` values coerced to `NaN` and imputed within the pipeline

Feature groups:
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Billing:** `MonthlyCharges`, `TotalCharges`

---

## 📁 Project Structure

```
churnsense-ai/
├── app.py                     # Streamlit web application (Demo & AutoML modes)
├── train_model.py             # Model training script -> models/ + reports/
├── requirements.txt           # Python dependencies (SHAP, scikit-learn, streamlit)
├── data/
│   └── WA_Fn-UseC_-_Telco_Customer_Churn.csv
├── src/
│   ├── data_loader.py         # Data loading, cleaning, validation & guards
│   ├── feature_engineering.py # Target separation & derived features
│   ├── pipeline.py            # Fit-once / serve-once pipeline builder
│   ├── explain.py             # SHAP XAI engine (individual & global explanations)
│   ├── automl.py              # Generic AutoML engine for custom CSV uploads
│   └── config.py              # Configuration constants
├── models/
│   ├── churn_pipeline.pkl     # Serialized end-to-end model pipeline
│   └── metadata.json          # Model metrics, hyperparameters, timestamp
├── reports/
│   ├── model_comparison.csv   # Model comparison test metrics
│   └── feature_importance.csv # Permutation importance metrics
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory data analysis
│   └── 02_Model_Training.ipynb# Notebook training & evaluation workflow
├── service/                   # FastAPI REST scoring service (main.py)
├── tests/                     # Pytest suite (explain, automl, pipeline, API)
├── Dockerfile                 # Container image for FastAPI service
└── docs/                      # Documentation (QUICKSTART, MODEL_CARD, CONTRIBUTING)
```

---

## 🏗️ Architecture

The core architecture uses a **single pipeline object** that encapsulates all transformations:

```
Raw Data → FeatureEngineer → ColumnTransformer(Numeric: Impute+Scale | Categorical: Impute+OneHot) → Classifier
```

- **Fit Once, Serve Once**: Fitted once on training data and serialized to `models/churn_pipeline.pkl`.
- **Zero Train-Serve Skew**: Streamlit, batch scripts, and FastAPI call `.predict_proba()` directly on raw inputs.
- **AutoML Adaptability**: Generic `ColumnTransformer` handles arbitrary CSV structures with automatic dtype detection.

---

## 📸 Visuals

| Churn Rate by Contract | Model Comparison |
|:---:|:---:|
| ![churn by contract](screenshots/fig_churn_by_contract.png) | ![model comparison](screenshots/fig_model_comparison.png) |

| Confusion Matrix (Deployed Model) | Permutation Feature Importance |
|:---:|:---:|
| ![confusion matrix](screenshots/fig_confusion_matrix.png) | ![feature importance](screenshots/fig_feature_importance.png) |

---

## 🚀 Installation

```bash
git clone https://github.com/aditichougule/churnsense-ai.git
cd churnsense-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📖 Usage

### 1. Train the Telco Model
```bash
python train_model.py
```
Trains candidate models, performs 5-fold CV on train data, evaluates on the held-out test set, and updates `models/churn_pipeline.pkl`.

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.
- **🎯 Demo (Telco)**: Browse 5 prebuilt pages, analyze churn drivers, make predictions, and inspect SHAP explanations.
- **📤 Your own data**: Upload any CSV dataset, select target column, auto-detect features, train Logistic Regression & Random Forest models, review global SHAP importance, test individual predictions, and download the fitted model (`.pkl`).

### 3. Score a Customer in Python
```python
import joblib
import pandas as pd
from src.pipeline import FeatureEngineer  # Required for unpickling

pipeline = joblib.load("models/churn_pipeline.pkl")

customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
}

proba = pipeline.predict_proba(pd.DataFrame([customer]))[0, 1]
print(f"Predicted churn probability: {proba:.1%}")
```

---

## 🌐 API Service

FastAPI service (`service/main.py`) provides high-performance JSON scoring:

```bash
uvicorn service.main:app --reload          # Interactive docs at http://127.0.0.1:8000/docs
```

Or via Docker:
```bash
docker build -t churnsense-api .
docker run -p 8000:8000 churnsense-api
```

Endpoints:
- `GET /health` - Service health & model load status
- `GET /model` - Model metadata & evaluation metrics
- `POST /predict` - Single customer churn scoring
- `POST /predict/batch` - Batch scoring (up to 1,000 records)

Sample request:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "gender":"Male","SeniorCitizen":1,"Partner":"No","Dependents":"No","tenure":1,
  "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
  "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
  "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "MonthlyCharges":95.0,"TotalCharges":190.0
}'
```

---

## 📈 Model Performance

Evaluated on the held-out 20% test set (1,409 customers) using 5-fold CV on train data:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.731 | 0.496 | **0.797** | 0.611 | **0.847** |
| Random Forest | 0.771 | 0.553 | 0.727 | 0.628 | 0.840 |

✅ = deployed default model. Logistic Regression delivers the highest recall (0.80), ensuring maximum detection of at-risk customers.

---

## 🔍 Explainable AI (SHAP)

**ChurnSense AI** integrates SHAP (SHapley Additive exPlanations) across all workflows:
- **Individual Waterfall & Contribution Plots**: Show precisely how each feature pushes a customer's churn probability above or below the baseline.
- **Global Beeswarm & Importance Charts**: Identify macro trends and top risk drivers across the entire customer base.
- **Domain Feature Aggregation**: Translates complex one-hot encoded vectors back to business terms (e.g., combining `Contract_One year` and `Contract_Two year` into `Contract`).

---

## 🧪 Testing

Run unit and integration tests:

```bash
pytest -q
```

Test coverage includes:
- Pipeline serialization and fit-once / serve-once consistency
- SHAP explanation generation for linear and tree-based models
- AutoML data validation, automatic feature typing, and model training
- FastAPI endpoints, input schemas, and batch scoring
- Performance gates ensuring minimum ROC-AUC and recall thresholds

---

## 👤 Author & License

**Author:** Aditi Chougule  
**Project:** ChurnSense AI  
**Repository:** [churnsense-ai](https://github.com/aditichougule/churnsense-ai)  
**License:** [MIT License](LICENSE)
