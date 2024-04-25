import pytest
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.EDA import ProcessData, DataPipeline



@pytest.fixture
def transformer():
    return ProcessData()

def test_process_data_drop_nan(transformer):
    data = pd.DataFrame({'A': [1, 2, None, 4], 'B': ['a', 'b', 'c', 'd']})
    transformed_data = transformer.transform(data)
    assert transformed_data.shape == (3, 2)
    assert transformed_data.isnull().sum().sum() == 0  

def test_process_data_drop_empty_strings(transformer):
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', ' ', 'c', 'd']})
    transformed_data = transformer.transform(data)
    assert transformed_data.shape == (3, 2)
    assert not (transformed_data == ' ').any().any()  

def test_process_data_label_encoding(transformer):
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'd']})
    transformed_data = transformer.transform(data)
    assert transformed_data['B'].tolist() == [0, 1, 2, 3]
    assert transformed_data['A'].tolist() == [0, 1, 2, 3]  

def test_process_data_fit_returns_self(transformer):
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'd']})
    assert transformer.fit(data) is transformer
    assert transformer.transform(data) is not None  
    
    
    

class DataPipeline:
    def init(self, dataset_directory):
        self.dataset_directory = dataset_directory

    def process_data(self, df):
        X = df.drop('Target', axis=1)
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return X_train, X_test, y_train, y_test

@pytest.fixture
def data_pipeline():
    df = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [5, 4, 3, 2, 1],
        'Target': [0, 1, 0, 1, 0]
    })
    return DataPipeline(df)

def test_data_pipeline(data_pipeline):
    df = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [5, 4, 3, 2, 1],
        'Target': [0, 1, 0, 1, 0]
    })
    X_train, X_test, y_train, y_test = data_pipeline.process_data(df)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
