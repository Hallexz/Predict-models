import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pytest
from src.MLP import FeedForwardNN

# Define the data
data = {
    'Feature': ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
    'Label': ['Label1', 'Label2', 'Label1', 'Label2', 'Label1']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert categorical features to numerical
le = LabelEncoder()
df['Feature'] = le.fit_transform(df['Feature'])
df['Label'] = le.fit_transform(df['Label'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['Feature']], df['Label'], test_size=0.2, random_state=42)

# Convert the data to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

# Convert labels to categorical
y_train = to_categorical(y_train)

@pytest.fixture
def feed_forward_nn():
    return FeedForwardNN(input_dim=1, num_classes=2)  # input_dim should be equal to the number of features

def test_feed_forward_nn(feed_forward_nn):
    # Fit the model
    feed_forward_nn.fit(X_train, y_train)

    # Predict
    y_pred = feed_forward_nn.predict(X_train)
    assert y_pred.shape == y_train.shape

    # Score
    accuracy = feed_forward_nn.score(X_train, y_train)
    assert 0 <= accuracy <= 1
