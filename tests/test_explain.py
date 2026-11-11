"""Tests for the per-customer explanation (linear SHAP values)."""
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.explain import explain_customer, supports_explanation
from src.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureEngineer  # noqa: F401

PIPELINE_PATH = "models/churn_pipeline.pkl"

CUSTOMER = dict(
    gender="Male", SeniorCitizen=1, Partner="No", Dependents="No", tenure=1,
    PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
    OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
    StreamingTV="Yes", StreamingMovies="Yes", Contract="Month-to-month",
    PaperlessBilling="Yes", PaymentMethod="Electronic check",
    MonthlyCharges=95.0, TotalCharges=190.0,
)


@pytest.fixture(scope="module")
def pipeline():
    if not os.path.exists(PIPELINE_PATH):
        pytest.skip("Run `python train_model.py` to produce the pipeline first.")
    return joblib.load(PIPELINE_PATH)


def test_explanation_covers_source_features(pipeline):
    exp = explain_customer(pipeline, pd.DataFrame([CUSTOMER]))
    assert set(exp["feature"]) <= set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert len(exp) >= 1


def test_contributions_sum_to_logit(pipeline):
    """The decomposition must be exact: Σ contributions + intercept = logit."""
    if not supports_explanation(pipeline):
        pytest.skip("Deployed model is non-linear.")
    X = pd.DataFrame([CUSTOMER])
    exp = explain_customer(pipeline, X)
    intercept = pipeline.named_steps["classifier"].intercept_[0]
    logit = exp["contribution"].sum() + intercept
    prob_reconstructed = 1.0 / (1.0 + np.exp(-logit))
    prob_model = pipeline.predict_proba(X)[0, 1]
    assert abs(prob_reconstructed - prob_model) < 1e-6
