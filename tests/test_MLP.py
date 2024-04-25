from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest
from src.MLP import FeedForwardNN  


data = {
    'Feature': ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
    'Label': ['Label1', 'Label2', 'Label1', 'Label2', 'Label1']
}

# Создайте DataFrame
df = pd.DataFrame(data)

# Преобразуйте категориальные признаки в числовые
le = LabelEncoder()
df['Feature'] = le.fit_transform(df['Feature'])
df['Label'] = le.fit_transform(df['Label'])

# Разделите данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df[['Feature']], df['Label'], test_size=0.2, random_state=42)

# Преобразуйте данные в формат numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

@pytest.fixture
def feed_forward_nn():
    return FeedForwardNN(input_dim=1, num_classes=2)  # input_dim должен быть равен количеству признаков

def test_feed_forward_nn(feed_forward_nn):
    # Fit the model
    feed_forward_nn.fit(X_train, y_train)

    # Predict
    y_pred = feed_forward_nn.predict(X_train)
    assert y_pred.shape == y_train.shape

    # Score
    accuracy = feed_forward_nn.score(X_train, y_train)
    assert 0 <= accuracy <= 1
