"""
Cost Calculator for False-Decline Watchdog (real fraud data version).

Turns the model's risk probabilities into rupee cost estimates, so we can
see which decision threshold actually saves the merchant the most money on
real transactions - not just which threshold looks best in accuracy terms.
"""

import joblib
import pandas as pd

from train_model import DROP_COLUMNS, MODEL_PATH, TARGET_COLUMN, TEST_PATH

# ---------------------------------------------------------------------------
# Currency conversion - TransactionAmt in this dataset is in US dollars.
# Fixed rate for now; swap for a live exchange rate lookup later if needed.
# ---------------------------------------------------------------------------
USD_TO_INR_RATE = 83  # rupees per $1 (approximate)

# ---------------------------------------------------------------------------
# Cost assumptions - per the 2026 LexisNexis "True Cost of Fraud" study,
# every ~₹1 lost directly to a chargeback costs the merchant ~₹5.13 in total
# once card-network fees, penalties, and operational overhead are included.
# ---------------------------------------------------------------------------
CHARGEBACK_COST_MULTIPLIER = 5.13


def false_positive_cost(transaction_amt_usd):
    """
    Cost of wrongly flagging a real, honest customer as risky: the lost sale
    itself, converted to INR. This is a conservative/worst-case estimate -
    in reality, some wrongly-flagged customers get manually reviewed and
    still complete their purchase, so the true cost is often lower than this.
    """
    return transaction_amt_usd * USD_TO_INR_RATE


def false_negative_cost(transaction_amt_usd):
    """
    Cost of missing a real fraud transaction (letting it slip through and
    become a chargeback). Per the LexisNexis multiplier, the total cost is
    ~5.13x the transaction amount once fees/penalties/overhead are included.
    """
    transaction_amt_inr = transaction_amt_usd * USD_TO_INR_RATE
    return transaction_amt_inr * CHARGEBACK_COST_MULTIPLIER


# risk thresholds to compare - 0.5 is the "default" cutoff, the rest show how
# moving the threshold trades off false positives against false negatives
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def get_test_set_with_predictions():
    """
    Loads the already-prepared, time-split test set and scores it with the
    saved trained model to get each row's predicted fraud risk probability.
    """
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    pipeline = joblib.load(MODEL_PATH)
    risk_probability = pipeline.predict_proba(X_test)[:, 1]

    results = X_test.copy()
    results["actual_is_fraud"] = y_test.values
    results["risk_probability"] = risk_probability
    return results


def cost_at_threshold(results, threshold):
    """
    Flags every row as "predicted risky" if its risk_probability is at or
    above the threshold, then adds up:
      - false positive cost: flagged as risky, but was actually a genuine transaction
      - false negative cost: NOT flagged, but actually turned out to be fraud
    """
    predicted_risky = results["risk_probability"] >= threshold

    false_positives = results[predicted_risky & (results["actual_is_fraud"] == 0)]
    false_negatives = results[~predicted_risky & (results["actual_is_fraud"] == 1)]

    fp_cost = false_positive_cost(false_positives["TransactionAmt"]).sum()
    fn_cost = false_negative_cost(false_negatives["TransactionAmt"]).sum()
    total_cost = fp_cost + fn_cost

    return fp_cost, fn_cost, total_cost


def main():
    results = get_test_set_with_predictions()

    rows = [(t, *cost_at_threshold(results, t)) for t in THRESHOLDS]

    print(f"{'threshold':>9} | {'false pos. cost':>20} | {'false neg. cost':>20} | {'total cost':>18}")
    print("-" * 78)
    for threshold, fp_cost, fn_cost, total_cost in rows:
        print(f"{threshold:>9.1f} | Rs {fp_cost:>17,.2f} | Rs {fn_cost:>17,.2f} | Rs {total_cost:>15,.2f}")

    best_threshold, _, _, best_total = min(rows, key=lambda r: r[3])
    print(f"\nLowest total cost: threshold {best_threshold} (Rs {best_total:,.2f})")


if __name__ == "__main__":
    main()
