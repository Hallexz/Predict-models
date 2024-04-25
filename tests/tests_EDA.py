import pytest
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from src.EDA import ProcessData


class ProcessData (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.dropna()
        X = X[X.astype(str).ne(' ').all(axis=1)]
        le = LabelEncoder()
        for col in X.columns:
            X[col] = le.fit_transform(X[col])
        return X

def test_process_data_drop_nan():
    data = pd.DataFrame({'A': [1, 2, None, 4], 'B': ['a', 'b', 'c', 'd']})
    transformer = ProcessData()
    transformed_data = transformer.transform(data)
    assert transformed_data.shape == (3, 2)

def test_process_data_drop_empty_strings():
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', ' ', 'c', 'd']})
    transformer = ProcessData()
    transformed_data = transformer.transform(data)
    assert transformed_data.shape == (3, 2)

def test_process_data_label_encoding():
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'd']})
    transformer = ProcessData()
    transformed_data = transformer.transform(data)
    assert transformed_data['B'].tolist() == [0, 1, 2, 3]

def test_process_data_fit_returns_self():
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'd']})
    transformer = ProcessData()
    assert transformer.fit(data) is transformer

