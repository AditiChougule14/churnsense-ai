import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import engineer_features

def test_engineer_features_creates_total_services():
    # Setup dummy data
    data = {"OnlineSecurity": ["Yes", "No"], "StreamingTV": ["Yes", "No"]}
    df = pd.DataFrame(data)
    
    # Run function
    df_engineered, _ = engineer_features(df)
    
    # Assert
    assert "TotalServices" in df_engineered.columns
    assert df_engineered.loc[0, "TotalServices"] == 2
    assert df_engineered.loc[1, "TotalServices"] == 0

def test_contract_risk_mapping():
    data = {"Contract": ["Month-to-month", "One year", "Two year", "Unknown"]}
    df = pd.DataFrame(data)
    
    df_engineered, _ = engineer_features(df)
    
    assert "ContractRisk" in df_engineered.columns
    assert df_engineered.loc[0, "ContractRisk"] == 3.0
    assert df_engineered.loc[1, "ContractRisk"] == 2.0
    assert df_engineered.loc[2, "ContractRisk"] == 1.0
    assert df_engineered.loc[3, "ContractRisk"] == 2.0 # Default/fillna
