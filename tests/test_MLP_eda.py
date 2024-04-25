import pytest
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential
from keras.layers import Dense, Dropout
from src.MLP import FeedForwardNN  


@pytest.fixture
def feed_forward_nn():
    return FeedForwardNN(input_dim=10, num_classes=2)

def test_feed_forward_nn(feed_forward_nn):
    # Create some dummy data
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(2, size=(100, 2))

    # Fit the model
    feed_forward_nn.fit(X_train, y_train)

    # Predict
    y_pred = feed_forward_nn.predict(X_train)
    assert y_pred.shape == y_train.shape

    # Score
    accuracy = feed_forward_nn.score(X_train, y_train)
    assert 0 <= accuracy <= 1
