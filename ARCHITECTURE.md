# Architecture — False-Decline Watchdog

## 1. System overview

False-Decline Watchdog is a request/response system, not a real-time streaming pipeline. Four pieces connect in a straight line: raw transaction data is cleaned and engineered into a training-ready dataset by the **data pipeline**; that dataset trains a **model** which is saved to disk; a **FastAPI service** loads that saved model and exposes it as a single scoring endpoint; and a **static frontend** calls that endpoint directly from the browser to power a live demo. Nothing runs continuously — every score is computed fresh, on demand, in response to one request, using a model that was trained offline ahead of time.

## 2. Data flow

```
train_transaction.csv  ─┐
                         ├─► prepare_data.py ─► train_clean.csv
train_identity.csv     ─┘        │                   │
                         (clean, join,                ▼
                          engineer features,    train_model.py ─► trained_model.pkl
                          time-split)                  │              scaler.pkl
                                                        │       frequency_encoder.pkl
                                                        ▼                   │
                                                  test_clean.csv            │
                                                  (held-out eval)           ▼
                                                                     api/main.py
                                                                    (loads model,
                                                                     serves /score)
                                                                           │
                                                                           ▼
                                                                  frontend/index.html
                                                                (Try it live → POST /score)
```

The pipeline runs once, offline, to produce a trained model. The API loads that model once at startup and answers one request at a time. The frontend is a passive client — it has no logic of its own beyond translating simplified form fields into the API's real field names.

## 3. Component breakdown

**Data pipeline (`data/prepare_data.py`)** — Loads the raw IEEE-CIS transaction and identity files and left-joins them on `TransactionID`. Drops columns that are more than ~80% missing, keeps ~20 columns judged both informative and interpretable (transaction amount, product code, card attributes, address/distance fields, count features, device type), and engineers three additional features: `has_identity` (whether an identity record exists at all — a real predictive signal on its own), and cyclical `hour_of_day`/`day_of_week` derived from the transaction's time offset. Critically, it splits the data **chronologically** — earliest ~80% for training, most recent ~20% for testing — and fits all missing-value fill values (medians, etc.) on the training portion only, before applying them to the test portion, to avoid leaking future information into the training process.

**Model (`model/train_model.py`, `model/encoders.py`)** — A logistic regression classifier with `class_weight="balanced"`, chosen for its transparency: every coefficient is inspectable, which matters for a project whose whole premise is being honest about tradeoffs rather than opaque about them. High-cardinality ID-like columns (`card1`, `card2`, `card3`, `card5` — hundreds to thousands of distinct values with no real categorical meaning) are frequency-encoded via a custom `FrequencyEncoder` rather than one-hot encoded, which would otherwise blow up into thousands of sparse columns. Numeric features are scaled with `StandardScaler` before training. The model outputs a raw probability (0–1), not just a hard label — that probability is what everything downstream (the cost calculator, the API, the frontend) actually consumes.

**Cost logic (`model/cost_calculator.py`)** — Converts a risk decision into an estimated rupee cost. A false positive (a genuine customer wrongly flagged) is costed at the transaction amount itself — the lost sale. A false negative (real fraud missed) is costed at the transaction amount multiplied by 5.13, based on the 2026 LexisNexis *"True Cost of Fraud"* study, which found that every ₹1 of actual fraud costs a merchant roughly ₹5.13 in total once fees, penalties, and operational overhead are included. This module also computes total cost across a range of thresholds on the held-out test set, which is how the default threshold of 0.5 was chosen.

**API (`api/main.py`)** — A FastAPI service exposing a single `POST /score` endpoint. Loads the trained model, scaler, and frequency encoder once at startup. Validates every incoming request with Pydantic — required fields, positive transaction amounts, and enums restricted to the real category values the model was trained on (`ProductCD`, `card4`, `card6`), so invalid input is rejected before it ever reaches the model. The risk threshold used to decide `is_flagged` is a single named constant, overridable per request, rather than hardcoded logic — this is what lets the frontend's threshold slider work without a code change. CORS is enabled to allow the browser-based frontend to call the API directly.

**Frontend (`frontend/index.html`)** — A single, no-build-step HTML file with three sections: an overview/pitch, a live demo ("Try it live"), and real results ("The numbers"). The live demo is a genuine client of the API, not a mock — it sends real `POST /score` requests and renders the real response. Because the demo's simplified form fields (amount, merchant category, card type, a device/IP signal, a billing/shipping mismatch toggle) don't map one-to-one onto the model's raw feature set, the page includes an explicit translation layer converting them into the API's real field names and values (including currency conversion, since the model was trained on USD amounts). The threshold slider recomputes the flagged/cleared decision locally from the already-fetched risk score, avoiding redundant API calls.

## 4. Key design decisions and why

- **Real data over synthetic.** The project was initially built on a synthetic dataset, then deliberately rebuilt on the real IEEE-CIS Fraud Detection dataset — real transactions with fraud labels derived from real chargebacks — because a model's behavior on invented data doesn't reliably predict its behavior on real transactions, and the goal here was a workable system, not just a demo.
- **Time-based train/test split, not random.** Fraud patterns shift over time. A random split would let the model implicitly learn from transactions that occurred, chronologically, after the ones it's tested against — data leakage that overstates real-world performance. Splitting by time simulates the actual deployment scenario: scoring transactions the model has never seen, from after its training period ends.
- **`class_weight="balanced"`.** With fraud at only ~3.5% of transactions, an unweighted model learns that predicting "safe" for everyone is a cheap way to look accurate. Balancing the class weights forces the model to actually pay attention to the minority class.
- **Frequency encoding for high-cardinality card IDs.** `card1` alone has over 13,000 distinct values. One-hot encoding it would create thousands of near-useless sparse columns; frequency encoding captures how common each value is without that blowup.
- **Adjustable threshold, not a fixed one.** Where to draw the line between "flag this" and "clear this" changes the tradeoff between catching fraud and annoying real customers. That's treated explicitly as a business decision, not a fixed statistical output — the threshold is a parameter, not a constant baked into the model.
- **Reporting precision and recall honestly, together.** The project's central thesis is that most fraud tools advertise catch rate (recall) while hiding the cost of false positives. Precision (0.088) is reported alongside recall (0.612) everywhere in the project — the README, the frontend's "numbers" section — specifically so the tool doesn't repeat the pattern it's built to critique.
- **A real published cost multiplier, not a guessed one.** The 5.13x false-negative cost multiplier comes from a cited industry study rather than an invented placeholder, so the cost figures the tool reports are defensible, not decorative.

## 5. Known limitations

- **Low precision at the default threshold.** At threshold 0.5, only 8.8% of transactions the model flags are actually fraud — most flagged transactions are genuine customers. This is a direct, understood consequence of prioritizing recall on a rare-event problem, not an unexamined flaw, but it means the current threshold trades a lot of false alarms for its fraud-catch rate.
- **Transaction amount has weak influence in typical ranges.** The model's amount coefficient is real but small relative to other features (e.g. certain email-domain categories), so everyday amounts (hundreds to a few thousand rupees) move the score only slightly; the effect becomes visible mainly at unusually large amounts.
- **The demo form is a simplified proxy, not the full feature set.** A handful of anonymized identifier fields the model was trained on aren't meaningful for a person to enter by hand, so the frontend fills them with representative placeholder values rather than exposing every raw feature as a form field.
- **Not deployed.** The API runs locally via `uvicorn --reload` (a development server, not production-configured), and the frontend is a static file opened locally. Neither component is hosted anywhere yet.
