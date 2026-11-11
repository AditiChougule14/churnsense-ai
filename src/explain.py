"""
Explainable AI (XAI) utilities for customer churn prediction pipelines using SHAP.

Supports both:
1. Demo Telco pipeline (FeatureEngineer -> ColumnTransformer -> classifier)
2. Generic AutoML pipeline (ColumnTransformer -> classifier)

Provides:
- supports_explanation: Check if estimator can be explained with SHAP.
- explain_customer: Per-customer SHAP feature contributions aggregated to original features.
- explain_shap_global: Global mean absolute SHAP feature importance aggregated to original features.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import shap


def _original_feature_names(pipeline) -> list[str]:
    """Derive original feature names corresponding to each column of the preprocessed matrix.

    Inspects the fitted ColumnTransformer in ``pipeline.named_steps['preprocess']``.
    For one-hot encoded categorical columns, maps each generated dummy column back
    to its source categorical column name, reliably accounting for dropped or grouped categories.
    """
    preprocess = pipeline.named_steps["preprocess"]
    feature_names: list[str] = []

    for name, transformer, columns in preprocess.transformers_:
        if transformer == "drop":
            continue

        # Detect OneHotEncoder (can be directly a transformer or inside a sub-Pipeline)
        onehot = None
        if hasattr(transformer, "named_steps"):
            onehot = transformer.named_steps.get("onehot")
        elif hasattr(transformer, "categories_"):
            onehot = transformer

        if onehot is not None and hasattr(onehot, "categories_"):
            # Determine output feature counts per categorical column
            if hasattr(onehot, "_n_features_outs"):
                n_outs = onehot._n_features_outs
            elif hasattr(onehot, "_compute_n_features_outs"):
                n_outs = onehot._compute_n_features_outs()
            else:
                n_outs = []
                drop_idx = getattr(onehot, "drop_idx_", None)
                for i, cats in enumerate(onehot.categories_):
                    count = len(cats)
                    if drop_idx is not None and drop_idx[i] is not None:
                        count -= 1
                    n_outs.append(count)

            for col, count in zip(columns, n_outs):
                feature_names.extend([col] * count)
        else:
            # Numeric or passthrough transformers produce one output per input column
            feature_names.extend(list(columns))

    return feature_names


def supports_explanation(pipeline) -> bool:
    """Return True if the estimator in the pipeline supports SHAP explanations."""
    if not hasattr(pipeline, "named_steps") or "classifier" not in pipeline.named_steps:
        return False
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "coef_"):
        return True
    if hasattr(clf, "estimators_") or hasattr(clf, "tree_"):
        return True
    tree_classes = {
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
    }
    return clf.__class__.__name__ in tree_classes


def _prepare_background(
    pipeline, background_data: pd.DataFrame | None = None, nsamples: int = 100
) -> np.ndarray | None:
    """Transform a representative sample of raw background data into preprocessed format."""
    if background_data is None or len(background_data) == 0:
        return None

    try:
        sample_size = min(nsamples, len(background_data))
        bg_sample = background_data.sample(n=sample_size, random_state=42)
        X_bg = pipeline[:-1].transform(bg_sample)
        if hasattr(X_bg, "toarray"):
            X_bg = X_bg.toarray()
        return np.asarray(X_bg, dtype=float)
    except Exception as exc:
        logging.warning("Failed to prepare background data for SHAP: %s", exc)
        return None


def _compute_shap_values(
    pipeline,
    X_pre: np.ndarray,
    background_data: pd.DataFrame | None = None,
    n_bg_samples: int = 100,
) -> np.ndarray:
    """Compute SHAP values for preprocessed data X_pre.

    Returns a 2D numpy array of shape (n_samples, n_preprocessed_features)
    corresponding to the positive class (class 1) for binary classification.
    """
    clf = pipeline.named_steps["classifier"]
    X_bg = _prepare_background(pipeline, background_data, nsamples=n_bg_samples)

    if hasattr(clf, "coef_"):
        # Linear models (e.g., LogisticRegression)
        # If background is provided, use empirical background; otherwise use zero baseline
        # so feature contributions sum to (logit - intercept), preserving exact additivity.
        if X_bg is None or len(X_bg) == 0:
            X_bg = np.zeros((1, X_pre.shape[1]))
        explainer = shap.LinearExplainer(clf, X_bg)
        shap_vals = explainer(X_pre)
        vals = np.asarray(shap_vals.values)
    else:
        # Tree-based models (e.g., RandomForestClassifier)
        # Note: check_additivity=False prevents ExplainerError from floating-point
        # rounding differences common in tree ensemble probability outputs.
        explainer = None
        try:
            if X_bg is not None and len(X_bg) > 0:
                explainer = shap.TreeExplainer(clf, data=X_bg)
            else:
                explainer = shap.TreeExplainer(clf)
            try:
                shap_vals = explainer(X_pre, check_additivity=False)
            except TypeError:
                shap_vals = explainer(X_pre)
            vals = np.asarray(shap_vals.values)
        except Exception:
            try:
                explainer = shap.TreeExplainer(clf)
                try:
                    shap_vals = explainer(X_pre, check_additivity=False)
                except TypeError:
                    shap_vals = explainer(X_pre)
                vals = np.asarray(shap_vals.values)
            except Exception:
                if X_bg is not None and len(X_bg) > 0:
                    explainer = shap.Explainer(clf, X_bg)
                else:
                    explainer = shap.Explainer(clf)
                try:
                    shap_vals = explainer(X_pre, check_additivity=False)
                except TypeError:
                    shap_vals = explainer(X_pre)
                vals = np.asarray(shap_vals.values)

    # Handle binary classification class dimension: (n_samples, n_features, n_classes)
    if vals.ndim == 3:
        if vals.shape[2] == 2:
            vals = vals[:, :, 1]  # positive class (churn / class 1)
        else:
            vals = vals[:, :, -1]

    return vals


def explain_customer(
    pipeline,
    X_row: pd.DataFrame,
    background_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return per-feature contributions to churn prediction for a single customer.

    Args:
        pipeline: Fitted scikit-learn Pipeline (engineer -> preprocess -> classifier or preprocess -> classifier).
        X_row: Single-row raw customer DataFrame.
        background_data: Optional representative background DataFrame.

    Returns:
        DataFrame with columns ``feature`` and ``contribution``, sorted by absolute impact.
    """
    X_pre = pipeline[:-1].transform(X_row)
    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray()
    X_pre = np.asarray(X_pre, dtype=float)

    shap_matrix = _compute_shap_values(
        pipeline, X_pre, background_data=background_data, n_bg_samples=100
    )
    contributions = shap_matrix[0]

    feature_sources = _original_feature_names(pipeline)

    # In case feature length differs due to unexpected transformer structure
    if len(feature_sources) != len(contributions):
        try:
            preprocess = pipeline.named_steps["preprocess"]
            feature_sources = list(preprocess.get_feature_names_out())
        except Exception:
            feature_sources = [f"feature_{i}" for i in range(len(contributions))]

    df = pd.DataFrame({"feature": feature_sources, "contribution": contributions})
    agg = df.groupby("feature", as_index=False)["contribution"].sum()
    return (
        agg.reindex(agg["contribution"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )


def explain_shap_global(
    pipeline,
    X: pd.DataFrame,
    background_data: pd.DataFrame | None = None,
    nsamples: int = 200,
) -> pd.DataFrame:
    """Compute mean absolute SHAP feature importance for each original feature.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        X: Raw input DataFrame (e.g., sample of customers).
        background_data: Optional background DataFrame. If None, X is used as background source.
        nsamples: Maximum number of rows to evaluate for responsive dashboard performance.

    Returns:
        DataFrame with columns ``feature`` and ``importance`` (mean |SHAP|), sorted descending.
    """
    sample_size = min(nsamples, len(X))
    X_eval = X.sample(n=sample_size, random_state=42)

    X_pre = pipeline[:-1].transform(X_eval)
    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray()
    X_pre = np.asarray(X_pre, dtype=float)

    ref_bg = background_data if background_data is not None else X
    shap_matrix = _compute_shap_values(
        pipeline, X_pre, background_data=ref_bg, n_bg_samples=100
    )
    mean_abs_importance = np.mean(np.abs(shap_matrix), axis=0)

    feature_sources = _original_feature_names(pipeline)

    if len(feature_sources) != len(mean_abs_importance):
        try:
            preprocess = pipeline.named_steps["preprocess"]
            feature_sources = list(preprocess.get_feature_names_out())
        except Exception:
            feature_sources = [f"feature_{i}" for i in range(len(mean_abs_importance))]

    df = pd.DataFrame({"feature": feature_sources, "importance": mean_abs_importance})
    agg = df.groupby("feature", as_index=False)["importance"].sum()
    return (
        agg.reindex(agg["importance"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )
