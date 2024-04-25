from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
import glob
import os


class ProcessData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.dropna()
        X = X[X.astype(str).ne(' ').all(axis=1)]
        le = LabelEncoder()
        for col in X.columns:
            X[col] = le.fit_transform(X[col])
        return X


def find_dataset(file_path):
    return pd.read_csv(file_path)


class DataPipeline:
    def __init__(self, dataset_directory, process_data, find_dataset):
        self.dataset_directory = dataset_directory
        self.process_data = process_data
        self.find_dataset = find_dataset

    def process_data_pipeline(self):
        csv_files = glob.glob(f"{self.dataset_directory}/*.csv")
        processed_data = pd.concat([self.process_data(self.find_dataset(file)) for file in csv_files], ignore_index=True)

        X = processed_data.drop('Target', axis=1)
        y = processed_data['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return X_train, X_test, y_train, y_test


    
    



