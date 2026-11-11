"""API tests for the FastAPI scoring service (via Starlette TestClient)."""
from fastapi.testclient import TestClient

from service.main import app

client = TestClient(app)

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


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_shape():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn", "churn_probability", "risk"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk"] in {"low", "medium", "high"}


def test_high_risk_scores_higher_than_low_risk():
    high = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK).json()["churn_probability"]
    assert high > low + 0.20, f"high={high} not above low={low}"


def test_invalid_category_returns_422():
    bad = {**HIGH_RISK, "Contract": "Lifetime"}  # not an allowed value
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_out_of_range_returns_422():
    bad = {**HIGH_RISK, "tenure": -5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_predict():
    r = client.post("/predict/batch", json=[HIGH_RISK, LOW_RISK])
    assert r.status_code == 200
    out = r.json()
    assert len(out) == 2
    assert out[0]["churn_probability"] > out[1]["churn_probability"]
