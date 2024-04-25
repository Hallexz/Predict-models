import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from sklearn.base import BaseEstimator, TransformerMixin


class FeedForwardNN(BaseEstimator, TransformerMixin):  
    def __init__(self, input_dim, num_classes):
        self.model = self.build_model(input_dim, num_classes)

    def build_model(self, input_dim, num_classes):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(input_dim,)))  
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y=None):
        self.model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        loss, accuracy = self.model.evaluate(X, y)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return accuracy


 
