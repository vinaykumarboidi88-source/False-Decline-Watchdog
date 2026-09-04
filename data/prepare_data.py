"""
Builds the cleaned, ready-to-train dataset for False-Decline Watchdog from
the raw IEEE-CIS fraud detection files.

Steps: load only the columns we're keeping -> merge transaction + identity
-> engineer has_identity / hour_of_day / day_of_week -> split chronologically
into train/test -> fill in missing values (using train-only statistics, so
nothing from the "future" test period leaks into how we fill train) -> save.

Outputs: data/train_clean.csv, data/test_clean.csv
"""

import pandas as pd

TRANSACTION_PATH = "data/train_transaction.csv"
IDENTITY_PATH = "data/train_identity.csv"
TRAIN_OUTPUT_PATH = "data/train_clean.csv"
TEST_OUTPUT_PATH = "data/test_clean.csv"

TRAIN_FRACTION = 0.8  # earliest 80% of transactions (by time) go to training

# raw columns kept for v1 - everything else (the heavily-missing V/D/id_*
# columns) is simply never read in, rather than loaded and then dropped
TRANSACTION_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "P_emaildomain",
    "C1", "C2", "C13", "C14",
]

NUMERIC_COLUMNS_TO_IMPUTE = ["card2", "card3", "card5", "addr1", "addr2", "dist1"]
CATEGORICAL_COLUMNS_TO_FILL = ["card4", "card6", "P_emaildomain"]


def load_and_merge():
    print("Loading transaction data...")
    tx = pd.read_csv(TRANSACTION_PATH, usecols=TRANSACTION_COLUMNS)

    print("Loading identity data...")
    identity = pd.read_csv(IDENTITY_PATH, usecols=["TransactionID", "DeviceType"])

    # has_identity is computed from the raw join key, BEFORE the merge fills
    # in NaNs for DeviceType - this way we can tell "no identity record at
    # all" apart from "had a record but DeviceType wasn't captured in it"
    identity_ids = set(identity["TransactionID"])
    tx["has_identity"] = tx["TransactionID"].isin(identity_ids).astype(int)

    df = tx.merge(identity, on="TransactionID", how="left")
    return df


def engineer_time_features(df):
    # TransactionDT is seconds elapsed from an arbitrary reference point, not
    # a real calendar timestamp, so we only pull out the cyclical hour-of-day
    # and day-of-week patterns - never the raw/absolute value itself, which
    # would just let the model memorize "when" training happened to occur
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    return df


def time_based_split(df):
    # sort chronologically first - don't assume the file is already ordered
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    cutoff = df["TransactionDT"].quantile(TRAIN_FRACTION)
    train_df = df[df["TransactionDT"] <= cutoff].copy()
    test_df = df[df["TransactionDT"] > cutoff].copy()
    return train_df, test_df


def clean_missing_values(train_df, test_df):
    """
    Fills missing values using statistics computed from the TRAINING portion
    only, then applies those same values to both train and test. Computing
    medians from the full dataset (including the "future" test period) would
    quietly leak information across the time split we just went to the
    trouble of creating.
    """
    # dist1 is ~60% missing but a known useful fraud signal, so we keep it
    # with an explicit "was this missing" flag instead of just imputing it away
    for df in (train_df, test_df):
        df["dist1_missing"] = df["dist1"].isna().astype(int)

    for col in NUMERIC_COLUMNS_TO_IMPUTE:
        fill_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(fill_value)
        test_df[col] = test_df[col].fillna(fill_value)

    # categorical columns: fill with an explicit "missing" label rather than
    # guessing a value, so the model can treat "we don't know" as its own case
    for col in CATEGORICAL_COLUMNS_TO_FILL:
        train_df[col] = train_df[col].fillna("missing")
        test_df[col] = test_df[col].fillna("missing")

    # DeviceType: distinguish "no identity record at all" (has_identity == 0)
    # from "had an identity record but device type wasn't captured in it"
    for df in (train_df, test_df):
        df["DeviceType"] = df["DeviceType"].where(df["has_identity"] == 1, other="no_identity")
        df["DeviceType"] = df["DeviceType"].fillna("unknown_device")

    return train_df, test_df


def main():
    df = load_and_merge()
    df = engineer_time_features(df)

    train_df, test_df = time_based_split(df)
    train_df, test_df = clean_missing_values(train_df, test_df)

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print(f"\nSaved {len(train_df)} rows to {TRAIN_OUTPUT_PATH}")
    print(f"Saved {len(test_df)} rows to {TEST_OUTPUT_PATH}")
    print(f"\nColumns ({len(train_df.columns)}): {list(train_df.columns)}")
    print(f"\nTrain fraud rate: {train_df['isFraud'].mean() * 100:.3f}%")
    print(f"Test fraud rate:  {test_df['isFraud'].mean() * 100:.3f}%")


if __name__ == "__main__":
    main()
