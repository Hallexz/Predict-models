import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pytest
from src.MLP import FeedForwardNN


data = {
    'Feature': ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
    'Label': ['Label1', 'Label2', 'Label1', 'Label2', 'Label1']
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['Feature'] = le.fit_transform(df['Feature'])
df['Label'] = le.fit_transform(df['Label'])

X_train, X_test, y_train, y_test = train_test_split(df[['Feature']], df['Label'], test_size=0.2, random_state=42)

X_train = np.array(X_train)
y_train = np.array(y_train)

y_train = to_categorical(y_train)

@pytest.fixture
def feed_forward_nn():
    return FeedForwardNN(input_dim=1, num_classes=2)  

def test_feed_forward_nn(feed_forward_nn):
    feed_forward_nn.fit(X_train, y_train)

    y_pred = feed_forward_nn.predict(X_train)
    assert y_pred.shape == y_train.shape

    accuracy = feed_forward_nn.score(X_train, y_train)
    assert 0 <= accuracy <= 1
