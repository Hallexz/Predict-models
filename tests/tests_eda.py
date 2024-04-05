import numpy as np

from alg_eda import mykmeans
from src.eda import process_data, perform_pca, y_data, preprocess_and_split

import pytest
import os


@pytest.fixture(scope="module")
def setup_data():
    files = os.listdir()
    csv_files = [f for f in files if f.endswith('.csv')]
    dataframes = {}
    my_kmeans = mykmeans(random_state=0)
    for file in csv_files:
        df = process_data(file)
        similar_columns, _ = perform_pca(df)
        df = my_kmeans.merge_similar_columns(df, similar_columns)
        y = y_data(df)
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        X_train, X_test, y_train, y_test = preprocess_and_split(df, y, numerical_columns, categorical_columns)
        dataframes[file] = (X_train, X_test, y_train, y_test)
    return dataframes

def test_process_data(setup_data):
    dataframes = setup_data
    for file, data in dataframes.items():
        X_train, X_test, y_train, y_test = data
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
