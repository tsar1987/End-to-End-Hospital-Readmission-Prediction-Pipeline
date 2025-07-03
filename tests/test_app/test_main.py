# File: tests/test_main.py

from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_predict_endpoint_success():
    """Test sending valid data and expecting a successful prediction."""
    valid_payload = {
        "race": "Caucasian", "gender": "Female", "age": "[70-80)",
        "admission_type_id": "1", "discharge_disposition_id": "1", "admission_source_id": "7",
        "time_in_hospital": "5", "num_lab_procedures": "40", "num_procedures": "1",
        "num_medications": "15", "number_outpatient": "0", "number_emergency": "0",
        "number_inpatient": "0", "number_diagnoses": "8", "diag_1": "428",
        "diag_2": "411", "diag_3": "250", "max_glu_serum": "None", "A1Cresult": ">7",
        "metformin": "No", "repaglinide": "No", "nateglinide": "No", "chlorpropamide": "No",
        "glimepiride": "No", "acetohexamide": "No", "glipizide": "No", "glyburide": "No",
        "tolbutamide": "No", "pioglitazone": "No", "rosiglitazone": "No", "acarbose": "No",
        "miglitol": "No", "troglitazone": "No", "tolazamide": "No", "examide": "No",
        "citoglipton": "No", "insulin": "Steady", "glyburide-metformin": "No",
        "glipizide-metformin": "No", "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No", "metformin-pioglitazone": "No",
        "change": "Ch", "diabetesMed": "Yes"
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    response_json = response.json()
    assert "prediction_value" in response_json
    assert response_json["prediction_value"] in [0, 1]

def test_predict_endpoint_validation_error():
    """Test sending invalid data and expecting the API to reject it."""
    invalid_payload = {"race": "Caucasian", "time_in_hospital": "five"} # 'five' is not an integer
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422 # 422 means Unprocessable Entity