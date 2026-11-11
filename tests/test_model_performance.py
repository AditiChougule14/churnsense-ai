"""Non-regression gate on model quality.

Reads the metrics persisted by the last training run and fails if the model
degrades below acceptable thresholds. This is the check that was missing: the
previous test suite stayed green while the shipped model had F1 = 0 and
ROC-AUC = 0.49 (it predicted "No Churn" for every customer).
"""
import json
import os

import pytest

METADATA_PATH = "models/metadata.json"

MIN_ROC_AUC = 0.78
MIN_RECALL = 0.45


@pytest.fixture(scope="module")
def test_metrics():
    if not os.path.exists(METADATA_PATH):
        pytest.skip("Run `python train_model.py` first to produce metadata.json.")
    with open(METADATA_PATH) as fh:
        return json.load(fh)["test_metrics"]


def test_roc_auc_above_threshold(test_metrics):
    assert test_metrics["roc_auc"] >= MIN_ROC_AUC, (
        f"ROC-AUC {test_metrics['roc_auc']} below {MIN_ROC_AUC}: the model may be "
        "broken or trained on bad/synthetic data."
    )


def test_recall_above_threshold(test_metrics):
    assert test_metrics["recall"] >= MIN_RECALL, (
        f"Recall {test_metrics['recall']} below {MIN_RECALL}: too many churners missed."
    )


def test_model_is_not_degenerate(test_metrics):
    # A majority-class-only classifier scores precision/recall/f1 == 0.
    assert test_metrics["f1"] > 0.0, "F1 is 0 — model predicts a single class."
