import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import pearsonr


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier


def process_data(df):
    df = df.fillna(value=0)
    df = df.drop_duplicates()
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
            df = df.astype('float64')
            return df



def perform_pca(df, n_components=2):
    my_pca = PCA(n_components=n_components)
    my_pca.fit(df)
    explained_variance_ratio = my_pca.explained_variance_ratio_
    print(f'Explained variance ratio: {explained_variance_ratio}')
    return my_pca, explained_variance_ratio


def y_data(df):
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=0)
    kmeans.fit(df)
    y = kmeans.labels_
    return y

def merge_similar_columns(df, similar_columns):
    valid_columns = set(df.columns)
    similar_columns = [(col1, col2) for col1, col2 in similar_columns if col1 in valid_columns and col2 in valid_columns]
    for col1, col2 in similar_columns:
        correlation = pearsonr(df[col1], df[col2])[0]
        if abs(correlation) >= 0.8:
            df[col1] = df[col1] + df[col2].median()
            df = df.drop(columns=col2)
            return df

def preprocess_and_split(df, y, numerical_columns, categorical_columns, test_size=0.2):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(), categorical_columns),
        ]
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_processed = pipeline.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def find_csv_files(directory_path='/data/raw/'):
    csv_files = []
    for root, directories, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.csv'):
                full_path = os.path.join(root, filename)
                csv_files.append(full_path)
    return csv_files


def get_categorical_and_numerical_columns(df):
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numerical_columns, categorical_columns


def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, classification_report
  
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
  
  

def visualize_pca(my_pca):

    plt.scatter(my_pca.components_[0, :], my_pca.components_[1, :])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()



def process_and_analyze_all_datasets(directory_path='/data/raw/'):
    csv_files = find_csv_files(directory_path)
    X_train = X_test = y_train = y_test = None  # Initialize the variables

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        df = process_data(df)
        my_pca, explained_variance_ratio = perform_pca(df)
        y = y_data(df)
        
        if y is not None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
            X_train, X_test, y_train, y_test = preprocess_and_split(df, y, numerical_columns, categorical_columns)
            visualize_pca(my_pca)
        else:
            print("y is None. Skipping preprocessing, splitting, and visualization.")

    return X_train, X_test, y_train, y_test
        
X_train, X_test, y_train, y_test = process_and_analyze_all_datasets()


def main():
    csv_file_path = '/data/raw/'

    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        df = process_data(df)
        y = y_data(df)
        numerical_columns, categorical_columns = get_categorical_and_numerical_columns(df)
        X_train, X_test, y_train, y_test = preprocess_and_split(df, y, numerical_columns, categorical_columns)

        if y_train is not None and y_test is not None:
            model1 = DecisionTreeClassifier()
            model2 = RandomForestClassifier(n_estimators=100)  

            ensemble_model = VotingClassifier(estimators=[('dt', model1), ('rf', model2)], voting='soft')
            ensemble_model.fit(X_train, y_train)
            evaluate_model(ensemble_model, X_test, y_test)
        else:
            print("y_train or y_test is None. Skipping model training and evaluation.")
    else:
        print(f"CSV file not found: {csv_file_path}")

if __name__ == "__main__":
    main()


    
    



