"""
Trains a logistic regression model to predict whether a transaction is
fraudulent (isFraud = 1), using the cleaned, already time-split IEEE-CIS
data produced by data/prepare_data.py.

Steps: load train_clean.csv / test_clean.csv -> encode categorical and
high-cardinality ID columns -> scale numeric columns -> train -> evaluate on
the held-out test set -> save the trained model (plus its scaler/encoder on
their own, for reuse or inspection later).
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from encoders import FrequencyEncoder

TRAIN_PATH = "data/train_clean.csv"
TEST_PATH = "data/test_clean.csv"
MODEL_PATH = "model/trained_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FREQUENCY_ENCODER_PATH = "model/frequency_encoder.pkl"

TARGET_COLUMN = "isFraud"

# columns we don't train on: TransactionID is just a row identifier, and
# TransactionDT is the raw time counter we already converted into
# hour_of_day / day_of_week during data prep - keeping the raw value around
# would let the model "memorize" specific moments in time instead of
# learning patterns that generalize to new transactions
DROP_COLUMNS = ["TransactionID", "TransactionDT"]

# ordinary categories - few enough distinct values that one-hot encoding
# makes sense (each value becomes its own meaningful yes/no column)
LOW_CARDINALITY_COLUMNS = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]

# card1/card2/card3/card5 are anonymized card/BIN identifiers with hundreds
# to thousands of distinct values (card1 alone has 13,000+). One-hot encoding
# these would create thousands of mostly-empty columns, and there isn't a
# real "category" meaning to them anyway - they're IDs, not labels. Instead
# we frequency-encode them: each value is replaced by how often it appeared
# in the training data, so "this is a very common card" vs. "this exact card
# value has barely been seen before" becomes one clean numeric signal -
# which matters for fraud, since fraudulent transactions often use less
# common / newer card identifiers.
HIGH_CARDINALITY_ID_COLUMNS = ["card1", "card2", "card3", "card5"]

# plain numeric columns, including the 0/1 engineered flags - scaling a
# binary flag alongside the others doesn't hurt anything, it just keeps the
# preprocessing code simple (one scaler for every number-like column)
NUMERIC_COLUMNS = [
    "TransactionAmt", "addr1", "addr2", "dist1", "dist1_missing",
    "C1", "C2", "C13", "C14", "has_identity", "hour_of_day", "day_of_week",
]


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def build_pipeline():
    """
    A Pipeline bundles preprocessing + the model together, so later (in the
    API) we can just call pipeline.predict() on raw input without redoing
    the encoding by hand.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("category_encoder", OneHotEncoder(handle_unknown="ignore"), LOW_CARDINALITY_COLUMNS),
            ("id_frequency_encoder", FrequencyEncoder(), HIGH_CARDINALITY_ID_COLUMNS),
            ("scaler", StandardScaler(), NUMERIC_COLUMNS),
        ],
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        # class_weight="balanced" tells the model to pay more attention to
        # the minority "fraud" class instead of defaulting to "not fraud" for
        # almost everything, since fraud is only ~3.5% of this data.
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    return pipeline


def main():
    X_train, X_test, y_train, y_test = load_data()

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # evaluate on the held-out (most recent, time-wise) test set only
    y_pred = pipeline.predict(X_test)

    # predict_proba returns [probability of 0, probability of 1] per row;
    # we only want the probability of "1" (fraud) as the risk score
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print()
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"                   predicted 0   predicted 1")
    print(f"actual 0 (safe)    {cm[0][0]:>11}   {cm[0][1]:>11}")
    print(f"actual 1 (fraud)   {cm[1][0]:>11}   {cm[1][1]:>11}")

    # build a small table of test rows alongside their predicted risk
    # probability, so we can sanity-check what the score looks like next to
    # real transaction details
    results = X_test.copy()
    results["actual_is_fraud"] = y_test.values
    results["risk_probability"] = y_proba.round(4)

    print("\nSample rows with predicted risk probability:")
    sample_columns = ["TransactionAmt", "ProductCD", "card4", "actual_is_fraud", "risk_probability"]
    print(results[sample_columns].head(10).to_string(index=False))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")

    # also save the fitted scaler and frequency encoder on their own, since
    # they'll need to be reused consistently if we ever score new
    # transactions outside of this pipeline object
    fitted_preprocessor = pipeline.named_steps["preprocessing"].named_transformers_
    joblib.dump(fitted_preprocessor["scaler"], SCALER_PATH)
    joblib.dump(fitted_preprocessor["id_frequency_encoder"], FREQUENCY_ENCODER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved frequency encoder to {FREQUENCY_ENCODER_PATH}")


if __name__ == "__main__":
    main()
