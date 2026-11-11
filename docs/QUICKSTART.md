
# 🚀 ChurnSense AI - Quick Start Guide

> **Predict Churn. Understand Why. Drive Retention.**

Welcome to **ChurnSense AI**! This guide will help you get up and running quickly with model training, the interactive Streamlit dashboard, explainability features, and the API service.

**Author:** Aditi Chougule  
**Repository:** [churnsense-ai](https://github.com/aditichougule/churnsense-ai)

---

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Telco Benchmark Model

```bash
python train_model.py
```

This will:

- Load and validate the Telco dataset.
- Engineer features inside the fit-once pipeline.
- Train and evaluate Logistic Regression and Random Forest.
- Select the best model via 5-fold cross-validation.
- Save `models/churn_pipeline.pkl` and `models/metadata.json`.
- Generate reports in `reports/`.

**Expected output:**

```text
================================================================================
CHURNSENSE AI - MODEL TRAINING PIPELINE
================================================================================
[1/6] Loading and cleaning dataset...
[2/6] Splitting train/test (stratified)...
[3/6] Selecting model via 5-fold CV (ROC-AUC) on train...
[4/6] Evaluating candidates on the held-out test set...
[5/6] Serializing the best model and metadata...
[6/6] Computing permutation feature importance...
✓ TRAINING COMPLETED SUCCESSFULLY
================================================================================
```

### 3. Run the Web Application

```bash
streamlit run app.py
```

The application will launch in your browser at:

`http://localhost:8501`

---

## 📁 Project Structure

```text
churnsense-ai/
├── app.py                      # Streamlit application (Demo & AutoML modes)
├── train_model.py              # Model training script
├── README.md                   # Comprehensive documentation
├── requirements.txt            # Python dependencies
├── data/                       # Dataset directory
│   └── WA_Fn-UseC_-_Telco_Customer_Churn.csv
├── src/                        # Core ML modules
│   ├── data_loader.py          # Data loading, cleaning, validation
│   ├── feature_engineering.py  # Target separation + derived features
│   ├── pipeline.py             # Fit-once / serve-once pipeline
│   ├── explain.py              # SHAP Explainable AI (individual & global)
│   ├── automl.py               # AutoML engine for custom CSV files
│   └── config.py               # Configuration constants
├── models/                     # Serialized models and metadata
│   ├── churn_pipeline.pkl      # Fitted pipeline
│   └── metadata.json           # Metrics and metadata
├── reports/                    # Performance reports and importance metrics
└── service/                    # FastAPI scoring service (main.py)
```

---

## 🎯 Exploring the Web Application

The sidebar allows you to select between two operational modes:

### Mode 1: 🎯 Demo (Telco)

Explore the pre-trained benchmark model across five dedicated views:

1. **🏠 Overview:** Business context, dataset summary metrics, and platform navigation.
2. **📈 Churn Insights:** Interactive distributions, churn rate by contract, internet service, and billing factors.
3. **🎯 Make Prediction:** Enter customer details, obtain instant churn probability and risk tier, view retention advice, and inspect individual SHAP feature contributions.
4. **📋 Model Details:** Review confusion matrix, ROC-AUC, classification metrics, and global SHAP feature importance.
5. **📊 Data Overview:** Explore raw data, missing value diagnostics, and feature statistics.

### Mode 2: 📤 Your Own Data (AutoML)

Train a churn model on custom business data:

1. Upload any CSV file.
2. Select your target variable (binary churn column).
3. Optionally exclude identifier columns (or let auto-detection exclude them).
4. Click **Train & Compare Models** to train both Logistic Regression and Random Forest.
5. Inspect cross-validation results, model metrics, and global SHAP feature importance.
6. Enter custom values in the interactive prediction form and see individual SHAP explanations.
7. Download the trained model `.pkl` for downstream deployment.

---

## 🤖 Model Information & Evaluation

### Supported Models

1. **Logistic Regression:** High recall, transparent linear coefficients, fast inference.
2. **Random Forest:** Ensemble decision trees, non-linear interactions, robust performance.

### Key Metrics

- **Recall:** Proportion of true churners captured (prioritized to maximize retention).
- **ROC-AUC:** Overall ranking ability across all probability thresholds.
- **Precision:** Proportion of flagged customers who actually churn.
- **F1-Score:** Harmonic mean of precision and recall.

---

## 🔍 Explainable AI with SHAP

**ChurnSense AI** includes built-in SHAP explanations:

- **Individual SHAP Contribution Plot:** Explains individual customer predictions by showing how each attribute contributes to the prediction.
- **Global Feature Importance:** Highlights the top drivers of churn across the entire customer base.
- **Feature Aggregation:** Dynamically aggregates one-hot encoded categories into intuitive feature-level impacts.

---

## 🌐 Running the FastAPI Service

To serve predictions via HTTP:

```bash
uvicorn service.main:app --reload
```

Open interactive Swagger documentation at:

`http://127.0.0.1:8000/docs`

---

## 🧪 Running Tests

Verify your environment with pytest:

```bash
pytest -q
```

---

## 📞 Getting Help & Resources

- **GitHub Repository:** [churnsense-ai](https://github.com/aditichougule/churnsense-ai)
- **Author:** Aditi Chougule
- **Documentation:** Check `README.md` and `docs/MODEL_CARD.md`.
