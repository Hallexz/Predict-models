from sklearn.pipeline import Pipeline

from MLP import FeedForwardNN
from EDA import ProcessData, find_dataset, DataPipeline



dataset_directory = '/home/hallex/spyd/study/Predict-models/data/raw'
pipeline = DataPipeline(dataset_directory, ProcessData().transform, find_dataset)
X_train, X_test, y_train, y_test = pipeline.process_data_pipeline()

process_data = ProcessData()
feed_forward_nn = FeedForwardNN(input_dim=X_train.shape[1], num_classes=y_train.shape[1])

pipeline = Pipeline([
    ('process_data', process_data),
    ('feed_forward_nn', feed_forward_nn)
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
