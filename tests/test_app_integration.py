"""End-to-end checks on the serving pipeline.

These replace the old shape-only test (which passed even when the model ignored
its inputs). They assert real behaviour: single-row inference works, it is
deterministic, and a high-risk profile scores higher than a low-risk one — a
direct guard against the historical train/serve skew.
"""
import os
import sys

import joblib
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import FeatureEngineer  # noqa: F401  (needed to unpickle the pipeline)

PIPELINE_PATH = "models/churn_pipeline.pkl"

LOW_RISK = dict(
    gender="Female", SeniorCitizen=0, Partner="Yes", Dependents="Yes", tenure=72,
    PhoneService="Yes", MultipleLines="Yes", InternetService="DSL",
    OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes",
    TechSupport="Yes", StreamingTV="No", StreamingMovies="No",
    Contract="Two year", PaperlessBilling="No",
    PaymentMethod="Credit card (automatic)", MonthlyCharges=25.0, TotalCharges=1800.0,
)
HIGH_RISK = dict(
    gender="Male", SeniorCitizen=1, Partner="No", Dependents="No", tenure=1,
    PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
    OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No",
    TechSupport="No", StreamingTV="Yes", StreamingMovies="Yes",
    Contract="Month-to-month", PaperlessBilling="Yes",
    PaymentMethod="Electronic check", MonthlyCharges=105.0, TotalCharges=105.0,
)


@pytest.fixture(scope="module")
def pipeline():
    if not os.path.exists(PIPELINE_PATH):
        pytest.skip("Run `python train_model.py` to produce the pipeline artifact first.")
    return joblib.load(PIPELINE_PATH)


def test_single_row_inference_runs(pipeline):
    proba = pipeline.predict_proba(pd.DataFrame([LOW_RISK]))
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_inference_is_deterministic(pipeline):
    p1 = pipeline.predict_proba(pd.DataFrame([HIGH_RISK]))[0, 1]
    p2 = pipeline.predict_proba(pd.DataFrame([HIGH_RISK]))[0, 1]
    assert p1 == p2


def test_model_actually_uses_inputs(pipeline):
    """Regression guard: a high-risk profile must score clearly higher than a
    low-risk one. If inputs were being silently dropped (the old skew bug), the
    two probabilities would be nearly identical."""
    low = pipeline.predict_proba(pd.DataFrame([LOW_RISK]))[0, 1]
    high = pipeline.predict_proba(pd.DataFrame([HIGH_RISK]))[0, 1]
    assert high > low + 0.20, f"high={high:.3f} not meaningfully above low={low:.3f}"
