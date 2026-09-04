"""
Custom encoders shared by train_model.py and cost_calculator.py.

Kept in their own module (rather than defined inline in train_model.py) so
that when the trained pipeline is saved with joblib/pickle, it can always be
reloaded from any other script - pickle needs to import the exact class
definition by module name, and a class defined in a "__main__" script isn't
importable from anywhere else.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces each category value with how often it appeared in the training
    data (as a proportion between 0 and 1). Learns the frequencies from
    whatever it's fit on (the training set only), then reuses those exact
    frequencies at prediction time - a value never seen during training maps
    to 0, i.e. treated as maximally unusual.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.freq_maps_ = {col: X[col].value_counts(normalize=True) for col in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        encoded = np.column_stack([
            X[col].map(self.freq_maps_[col]).fillna(0.0).to_numpy(dtype=float)
            for col in self.columns_
        ])
        return encoded
