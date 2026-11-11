"""Tests for the 'bring your own data' AutoML-lite trainer."""
import numpy as np
import pandas as pd
import pytest

from src.automl import DataValidationError, prepare_target, train_on_dataframe


def _synthetic(n=400, seed=0):
    """A small dataset with real signal, an ID-like column and a categorical."""
    rng = np.random.default_rng(seed)
    plan = rng.choice(["basic", "pro", "premium"], n)
    monthly = rng.normal(50, 15, n).round(2)
    logit = -2 + 0.04 * monthly + (plan == "basic") * 1.5 - (plan == "premium") * 1.0
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],  # ID-like -> dropped
        "age": rng.integers(18, 80, n),
        "plan": plan,
        "monthly_charge": monthly,
        "Churn": churn,
    })


def test_trains_and_learns_signal():
    res = train_on_dataframe(_synthetic(), "Churn")
    assert res.best_model in res.results
    assert res.results[res.best_model]["metrics"]["roc_auc"] > 0.6
    assert "customer_id" in dict(res.dropped)         # ID-like column dropped
    assert "monthly_charge" in res.numeric
    assert "plan" in res.categorical
    assert set(res.schema) == set(res.numeric + res.categorical)


def test_downloadable_pipeline_scores_a_row():
    res = train_on_dataframe(_synthetic(), "Churn")
    row = {"age": 40, "plan": "basic", "monthly_charge": 90.0}
    proba = res.best_pipeline.predict_proba(pd.DataFrame([row]))[0, 1]
    assert 0.0 <= proba <= 1.0


def test_too_few_rows():
    with pytest.raises(DataValidationError):
        train_on_dataframe(_synthetic(n=20), "Churn")


def test_non_binary_target_rejected():
    with pytest.raises(DataValidationError):
        train_on_dataframe(_synthetic(), "age")


def test_prepare_target_picks_positive_class():
    y, pos = prepare_target(pd.Series(["No", "Yes", "No", "Yes", "No"]))
    assert pos == "Yes"
    assert list(y) == [0, 1, 0, 1, 0]
