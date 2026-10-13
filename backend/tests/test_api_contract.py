from __future__ import annotations
import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd
from backend.main import app

client = TestClient(app)


def test_production_prediction_with_missing_values():
    """Test single-row prediction with NaN values using the connected fitted preprocessor."""
    payload = {
        "features": {
            "ph": None,
            "Hardness": 200.0,
            "Solids": None,
            "Chloramines": 5.0,
            "Sulfate": 300.0,
            "Conductivity": 400.0,
            "Organic_carbon": 10.0,
            "Trihalomethanes": None,
            "Turbidity": 3.0,
        },
        "source": "api_test"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "model" in data["model_name"].lower()


def test_production_batch_prediction_with_missing_values():
    """Test file prediction containing NaN values."""
    csv_content = (
        "ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity\n"
        "7.2,210.0,18000.0,4.5,290.0,500.0,12.0,60.0,4.1\n"
        ",205.0,,4.8,295.0,510.0,12.5,,4.3\n"
    )
    files = {"file": ("batch_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/predict-file", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["row_count"] == 2
    assert len(data["results"]) == 2
    for item in data["results"]:
        assert item["prediction"] in [0, 1]


def test_preprocessing_status_reporting():
    """Verify that GET /api/preprocessing/status reports artifact status honestly."""
    response = client.get("/api/preprocessing/status")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["artifact_exists"] is True
    assert data["production_model_uses_preprocessing"] is True
    assert data.get("pipeline_name") is not None


def test_device_exact_code_lookup_vs_numeric_id():
    """Verify device code lookup matches exact device_code first before integer id."""
    # Register device with numeric code "9999"
    reg1 = client.post("/api/devices", json={
        "device_id": "9999",
        "device_name": "Numeric Code Sensor",
        "location": "Station A"
    })
    assert reg1.status_code in [200, 400, 409]

    # Fetch device status using exact device code "9999"
    status_resp = client.get("/api/devices/9999")
    assert status_resp.status_code == 200
    assert status_resp.json()["device_id"] == "9999"


def test_dataset_compatibility_and_readiness():
    """Verify that dataset inspection differentiates schema compatibility and prediction readiness."""
    csv_content = (
        "ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity\n"
        "7.0,200.0,15000.0,4.0,280.0,500.0,11.0,55.0,3.5\n"
    )
    files = {"file": ("compat_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    upload_resp = client.post("/api/datasets/upload", files=files)
    assert upload_resp.status_code == 200
    data = upload_resp.json()
    assert "prediction_readiness" in data
    assert data["prediction_readiness"]["schema_compatible"] is True
    assert data["prediction_readiness"]["prediction_ready"] is True


def test_classification_with_numeric_target():
    """Verify classification accepts numeric target representing discrete classes without 422."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "target_num": [0, 0, 0, 0, 1, 1, 1, 1]
    })
    from backend.services.classification_service import run_classification
    res = run_classification(
        df,
        dataset_id="test_num_target",
        dataset_metadata={},
        target="target_num",
        features=["feature1"],
        algorithm="decision-tree"
    )
    assert res["complete_rows"] == 8
    assert "accuracy" in res["metrics"]
