"""
False-Decline Watchdog API (real fraud data version).

One endpoint, /score, that takes a transaction's details and returns:
- a risk score (probability it's fraud)
- whether it's flagged, using the configurable RISK_THRESHOLD below
- the rupee cost if this specific prediction turns out wrong
- a short plain-English reason for the call
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# model/ and api/ are sibling folders, so we add model/ to the import path to
# reuse the exact same cost-of-being-wrong logic defined in cost_calculator.py
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
sys.path.insert(0, str(MODEL_DIR))

from cost_calculator import false_negative_cost, false_positive_cost  # noqa: E402
# FrequencyEncoder isn't called directly here, but the saved pipeline was
# pickled with a reference to this exact class - joblib needs it importable
# under the same module name (encoders) to load the pipeline back in.
from encoders import FrequencyEncoder  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Risk threshold - deliberately kept as a single, easy-to-find constant
# (not buried in the scoring logic) so it can be changed quickly, e.g. to
# show different cutoffs live in a demo. It can also be overridden per
# request - see OrderRequest.threshold below - without restarting the server.
# ---------------------------------------------------------------------------
RISK_THRESHOLD = 0.5

# fallback used only when a caller doesn't provide dist1 - the median value
# from the training data, matching how missing dist1 was imputed during
# training (see data/prepare_data.py)
DIST1_MEDIAN_FALLBACK = 9.0

# the saved pipeline already bundles the one-hot encoder + frequency encoder
# + scaler + logistic regression model together, so it can score raw input
# directly.
model_pipeline = joblib.load(MODEL_DIR / "trained_model.pkl")

# also loaded on their own per project convention, even though the copies
# inside the pipeline above are what actually get used for scoring.
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
frequency_encoder = joblib.load(MODEL_DIR / "frequency_encoder.pkl")

app = FastAPI(title="False-Decline Watchdog")

# allows frontend/index.html (opened as a local file, a different origin
# from the API) to call this API directly from the browser during a demo -
# fine for a local demo tool, would need tightening for real deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# the known category codes actually present in the training data
# (data/train_clean.csv) - anything else gets rejected with a clear 422
# rather than silently passed to the model
PRODUCT_CODES = Literal["W", "C", "R", "H", "S"]
CARD_NETWORKS = Literal["visa", "mastercard", "american express", "discover", "missing"]
CARD_TYPES = Literal["debit", "credit", "charge card", "debit or credit", "missing"]


class OrderRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD; must be positive")
    ProductCD: PRODUCT_CODES
    card1: int
    card2: float
    card3: float
    card4: CARD_NETWORKS
    card5: float
    card6: CARD_TYPES
    addr1: float
    addr2: float
    P_emaildomain: str
    C1: float
    C2: float
    C13: float
    C14: float

    # optional: not every integration has distance/device data available.
    # When omitted, they're handled the same way missing values were
    # handled during training (see data/prepare_data.py).
    dist1: Optional[float] = None
    DeviceType: Optional[str] = None

    # optional per-request override of the default RISK_THRESHOLD, so the
    # same running server can demo different cutoffs without a restart
    threshold: Optional[float] = Field(default=None, ge=0, le=1)


class ScoreResponse(BaseModel):
    risk_score: float
    is_flagged: bool
    estimated_cost_if_wrong: float
    reason: str


def build_reason(order: OrderRequest, risk_score: float, is_flagged: bool, has_identity: int) -> str:
    """
    Small rule-based explanation, using the same signals the model was
    trained to pick up on. Not a full breakdown of the model's math, just a
    plain-English summary of what stands out about this transaction.
    """
    if is_flagged:
        risk_notes = []
        if order.TransactionAmt >= 500:
            risk_notes.append("high transaction value")
        if has_identity == 0:
            risk_notes.append("no device/identity data available")
        if order.ProductCD == "C":
            risk_notes.append("product category with historically higher fraud rate")
        if order.dist1 is None:
            risk_notes.append("billing/shipping distance unknown")

        detail = ", ".join(risk_notes) if risk_notes else "elevated overall risk score"
        return f"Flagged: {detail} (risk score {risk_score:.2f})"

    safe_notes = []
    if has_identity == 1:
        safe_notes.append("device/identity data available")
    if order.TransactionAmt < 100:
        safe_notes.append("low transaction value")
    if order.ProductCD == "W":
        safe_notes.append("product category with historically lower fraud rate")

    detail = ", ".join(safe_notes) if safe_notes else "low overall risk score"
    return f"Not flagged: {detail} (risk score {risk_score:.2f})"


@app.post("/score", response_model=ScoreResponse)
def score_order(order: OrderRequest) -> ScoreResponse:
    # has_identity mirrors training: 1 if identity/device data was provided
    # for this transaction, 0 otherwise
    has_identity = 1 if order.DeviceType is not None else 0
    device_type = order.DeviceType if has_identity == 1 else "no_identity"

    # dist1 is often unavailable at checkout time - fall back to the same
    # median used to fill missing values during training
    dist1_missing = 1 if order.dist1 is None else 0
    dist1 = order.dist1 if order.dist1 is not None else DIST1_MEDIAN_FALLBACK

    # hour_of_day/day_of_week were derived from TransactionDT during
    # training (a raw seconds-elapsed counter, not a real timestamp) to
    # capture daily/weekly behavior cycles. For a live request we use the
    # actual current time, which captures the same kind of cyclical pattern.
    now = datetime.now()
    hour_of_day = now.hour
    day_of_week = now.weekday()

    input_df = pd.DataFrame([{
        "TransactionAmt": order.TransactionAmt,
        "ProductCD": order.ProductCD,
        "card1": order.card1,
        "card2": order.card2,
        "card3": order.card3,
        "card4": order.card4,
        "card5": order.card5,
        "card6": order.card6,
        "addr1": order.addr1,
        "addr2": order.addr2,
        "dist1": dist1,
        "dist1_missing": dist1_missing,
        "P_emaildomain": order.P_emaildomain,
        "C1": order.C1,
        "C2": order.C2,
        "C13": order.C13,
        "C14": order.C14,
        "has_identity": has_identity,
        "DeviceType": device_type,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
    }])

    risk_score = float(model_pipeline.predict_proba(input_df)[:, 1][0])

    threshold = order.threshold if order.threshold is not None else RISK_THRESHOLD
    is_flagged = risk_score >= threshold

    # cost if this call turns out wrong: if we flagged them, the risk is that
    # they were actually a genuine transaction (false positive); if we
    # didn't flag them, the risk is that they were actually fraud (false negative)
    if is_flagged:
        estimated_cost_if_wrong = false_positive_cost(order.TransactionAmt)
    else:
        estimated_cost_if_wrong = false_negative_cost(order.TransactionAmt)

    reason = build_reason(order, risk_score, is_flagged, has_identity)

    return ScoreResponse(
        risk_score=round(risk_score, 4),
        is_flagged=is_flagged,
        estimated_cost_if_wrong=round(float(estimated_cost_if_wrong), 2),
        reason=reason,
    )
