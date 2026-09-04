# False-Decline Watchdog

False-Decline Watchdog is a fraud risk-scoring tool for e-commerce merchants, built for the Razorpay Buildathon (Track 2 — AI Risk Manager). Given a transaction's details, it returns a risk score and, just as importantly, the estimated rupee cost of that score being wrong in either direction — wrongly blocking a genuine customer, or missing a real fraud. Most fraud tools report only how much fraud they catch; this one is built around the idea that the cost of false positives deserves equal billing, because a wrongly declined customer is a real, measurable business loss too.

## The problem

Merchants lose money to fraud in two directions, not one. Fraudulent transactions turn into chargebacks, which cost the transaction amount plus card-network penalties and processing fees. But blocking too aggressively causes false declines — legitimate customers rejected at checkout — which is a lost sale with no offsetting benefit. Most fraud-detection tools are marketed purely on catch rate (recall), because a high catch rate is easy to advertise and the false-decline cost is invisible unless someone deliberately measures it. This project treats both costs as first-class outputs, not an afterthought.

## The data

This project uses the real **[IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection)** from Kaggle — 590,540 real e-commerce transactions with real chargeback-based fraud labels (`isFraud`), at a ~3.5% fraud rate. It is not synthetic data. `train_transaction.csv` (394 columns) is joined with `train_identity.csv` (41 columns) on `TransactionID`.

The data is split **chronologically** — the earliest ~80% of transactions (by `TransactionDT`) for training, the most recent ~20% for testing — rather than a random split. Fraud patterns shift over time, so a random split would let the model implicitly learn from transactions that, chronologically, happened after the ones it's being tested on (data leakage) and would overstate how well it would perform on genuinely new, future transactions. The time-based split simulates the real deployment scenario: scoring transactions the model has never seen, from a period after its training data ends.

## The approach

The model is a logistic regression classifier with `class_weight="balanced"` (fraud is rare, so the model needs help paying attention to the minority class), trained on ~20 selected real-world features — including `TransactionAmt`, `ProductCD`, the `card1`–`card6` card attributes, `addr1`/`addr2`, `dist1`, `C1`/`C2`/`C13`/`C14`, `DeviceType`, and two engineered features: `has_identity` (whether identity/device data was captured for the transaction at all — a genuinely predictive signal on its own) and cyclical `hour_of_day`/`day_of_week` features. High-cardinality ID-like columns (`card1`, `card2`, `card3`, `card5`) are frequency-encoded rather than one-hot encoded, since they have hundreds to thousands of distinct values with no real categorical meaning. The cost model is based on a real published industry figure — per the 2026 LexisNexis *"True Cost of Fraud"* study, every ₹1 lost to a chargeback costs the merchant ~₹5.13 in total once fees, penalties, and overhead are included — and the risk threshold used to decide `is_flagged` is a single adjustable value, overridable per API request.

## Results

On the held-out (chronologically later) test set of 118,108 transactions, at the default threshold of 0.5:

- **Precision: 0.088**
- **Recall: 0.612**

The model catches 61% of real fraud, at the cost of a low precision — most flagged transactions turn out to be genuine. This is an intentional, explainable consequence of `class_weight="balanced"` prioritizing fraud detection given a ~3.5% base rate, and the threshold is deliberately adjustable rather than fixed, since the "right" balance between catching fraud and avoiding false declines is a business decision, not a purely statistical one. See `model/cost_calculator.py` for how different thresholds translate into actual rupee cost tradeoffs.

## The frontend

`frontend/index.html` is a single-page, no-build-step site with three sections: **Overview** (the pitch and problem statement), **Try it live** (a real demo form wired to the actual `/score` API), and **The numbers** (the real results above, shown as stats). It supports a dark/light theme toggle.

The "Try it live" demo sends real requests directly from the browser to `POST http://127.0.0.1:8000/score` — it's calling the actual trained model, not a mock. Since the demo's simplified fields (amount, merchant category, card type, a device/IP risk signal, a billing/shipping mismatch toggle) don't map one-to-one onto the model's raw feature set, the page translates them into the API's real field names and values, and fills in the handful of anonymized identifier fields the model needs but that aren't meaningful to expose as inputs (documented in comments in the file). A threshold slider lets you see a transaction's flagged/cleared decision change live, without a new API call, since that's a pure comparison against the risk score already returned. If the API isn't running, the page shows a clear message telling you to start it, instead of failing silently.

One thing worth knowing if you're editing this file directly: the API has CORS enabled (`allow_origins=["*"]`) specifically so this page can call it from a browser regardless of how the page itself is loaded.

## Project structure

```
False-Decline Watchdog/
├── data/
│   ├── train_transaction.csv, train_identity.csv   # raw IEEE-CIS source data
│   ├── prepare_data.py                             # cleans, joins, engineers features, time-splits
│   └── train_clean.csv, test_clean.csv             # output of prepare_data.py, ready to train on
├── model/
│   ├── encoders.py                                 # custom FrequencyEncoder for high-cardinality ID columns
│   ├── train_model.py                              # trains and evaluates the logistic regression model
│   ├── cost_calculator.py                          # rupee cost-by-threshold analysis
│   └── trained_model.pkl, scaler.pkl, frequency_encoder.pkl  # saved model artifacts
├── api/
│   └── main.py                                     # FastAPI app exposing the /score endpoint (CORS enabled)
├── frontend/
│   └── index.html                                  # pitch page + live demo, calls /score directly from the browser
├── tests/
│   └── test_api.py                                 # automated tests for the API
├── requirements.txt
└── README.md
```

## How to run it

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the data preparation pipeline** (cleans and joins the raw CSVs, engineers features, splits chronologically):
   ```bash
   python data/prepare_data.py
   ```

3. **Train the model** (trains, evaluates on the held-out test set, saves the model/scaler/encoder):
   ```bash
   python model/train_model.py
   ```

4. *(Optional)* **Run the cost calculator** to see the rupee cost tradeoff across different risk thresholds:
   ```bash
   python model/cost_calculator.py
   ```

5. **Run the API**:
   ```bash
   uvicorn api.main:app --reload
   ```
   Then visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI, or `POST` to `/score` directly.

6. **Run the frontend**, with the API still running from step 5. Just open `frontend/index.html` directly in your browser (double-click it, or drag it into a tab) and try the "Try it live" tab — it calls your running API for real.

   If the page ever shows raw `{{ }}` text instead of real content (browser-dependent quirk, hasn't come up in normal testing), serve it over local HTTP instead as a fallback:
   ```bash
   cd frontend
   python -m http.server 8765
   ```
   then open `http://127.0.0.1:8765/index.html`.

7. **Run the tests**:
   ```bash
   python -m pytest tests/test_api.py -v
   ```

## Status / next steps

The backend (data pipeline, model, cost analysis, validated API) and frontend (pitch page + live demo) are both built and working together end-to-end. What's genuinely still open:

- **Deployment.** Everything currently runs locally — the API via `uvicorn --reload` (a dev server, not production-configured) and the frontend via a local static file server. Neither is hosted anywhere yet.
- **CORS is wide open** (`allow_origins=["*"]`) in `api/main.py`, which is fine for local development but should be tightened to a specific origin before any real deployment.
- The frontend's demo trades some fidelity for usability — a few real model features are auto-filled with representative values rather than exposed as form fields, since they're anonymized identifiers not meaningful for a person to enter by hand.
