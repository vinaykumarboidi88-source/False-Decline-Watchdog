"""
Automated tests for the /score endpoint (api/main.py).

Run with: python -m pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "api"))

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)

# a full, valid transaction used as the baseline for every test below -
# individual tests tweak just the field(s) they care about
VALID_ORDER = {
    "TransactionAmt": 250.0,
    "ProductCD": "W",
    "card1": 13553,
    "card2": 555.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 226.0,
    "card6": "debit",
    "addr1": 315.0,
    "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "C1": 1.0,
    "C2": 1.0,
    "C13": 1.0,
    "C14": 1.0,
    "dist1": 19.0,
    "DeviceType": "mobile",
}


def test_valid_request_returns_200_with_expected_fields():
    response = client.post("/score", json=VALID_ORDER)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"risk_score", "is_flagged", "estimated_cost_if_wrong", "reason"}
    assert isinstance(body["risk_score"], float)
    assert isinstance(body["is_flagged"], bool)
    assert isinstance(body["estimated_cost_if_wrong"], float)
    assert isinstance(body["reason"], str)
    assert 0.0 <= body["risk_score"] <= 1.0


def test_missing_required_field_returns_422():
    incomplete_order = VALID_ORDER.copy()
    del incomplete_order["TransactionAmt"]

    response = client.post("/score", json=incomplete_order)
    assert response.status_code == 422


def test_negative_transaction_amount_is_rejected():
    bad_order = {**VALID_ORDER, "TransactionAmt": -50.0}

    response = client.post("/score", json=bad_order)
    assert response.status_code == 422


def test_zero_transaction_amount_is_rejected():
    bad_order = {**VALID_ORDER, "TransactionAmt": 0}

    response = client.post("/score", json=bad_order)
    assert response.status_code == 422


def test_invalid_card4_value_is_rejected():
    bad_order = {**VALID_ORDER, "card4": "bitcoin"}  # not a real card network

    response = client.post("/score", json=bad_order)
    assert response.status_code == 422


def test_invalid_card6_value_is_rejected():
    bad_order = {**VALID_ORDER, "card6": "prepaid"}  # not a real card6 value

    response = client.post("/score", json=bad_order)
    assert response.status_code == 422


def test_all_real_card6_values_are_accepted():
    # card6 values actually seen in the training data (data/train_clean.csv) -
    # all of them should be valid input, not just "debit"/"credit"
    for card6_value in ["debit", "credit", "charge card", "debit or credit", "missing"]:
        order = {**VALID_ORDER, "card6": card6_value}
        response = client.post("/score", json=order)
        assert response.status_code == 200, f"card6={card6_value!r} should be accepted"


def test_threshold_override_changes_is_flagged():
    """
    Confirms the threshold override actually changes the flagging decision
    (not just accepted as input and ignored). Uses the model's own risk
    score for this order to pick a threshold just below it (should flag)
    and just above it (should not flag).
    """
    baseline_response = client.post("/score", json=VALID_ORDER)
    risk_score = baseline_response.json()["risk_score"]

    low_threshold = max(risk_score - 0.1, 0.0)
    high_threshold = min(risk_score + 0.1, 1.0)

    low_response = client.post("/score", json={**VALID_ORDER, "threshold": low_threshold})
    high_response = client.post("/score", json={**VALID_ORDER, "threshold": high_threshold})

    assert low_response.status_code == 200
    assert high_response.status_code == 200
    assert low_response.json()["is_flagged"] is True
    assert high_response.json()["is_flagged"] is False
