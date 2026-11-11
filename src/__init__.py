"""
ChurnSense AI - src package
Predict Churn. Understand Why. Drive Retention.

Author: Aditi Chougule
Repository: churnsense-ai

Modules:
- data_loader: dataset loading, cleaning and validation
- feature_engineering: target separation and derived features
- pipeline: the single fit-once / serve-once modeling pipeline
- explain: SHAP explainability for individual and global explanations
- automl: automated ML on custom CSV datasets
- config: configuration settings
"""

from .data_loader import load_telco_data, clean_data, validate_data, get_dataset_info
from .feature_engineering import separate_features_and_target, engineer_features
from .pipeline import (
    build_pipeline,
    add_engineered_features,
    FeatureEngineer,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from . import config

__version__ = "2.0.0"
__author__ = "Aditi Chougule"
__description__ = "ChurnSense AI - Predict Churn. Understand Why. Drive Retention."

__all__ = [
    # Data loading
    'load_telco_data',
    'clean_data',
    'validate_data',
    'get_dataset_info',
    # Feature engineering
    'separate_features_and_target',
    'engineer_features',
    # Pipeline
    'build_pipeline',
    'add_engineered_features',
    'FeatureEngineer',
    'NUMERIC_FEATURES',
    'CATEGORICAL_FEATURES',
    # Modules
    'config',
]
