import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from src.alg_eda import mypca, mykmeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


from sklearn.cluster import KMeans

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
    my_pca = mypca(n_components=n_components)
    similar_columns = my_pca.fit(df)
    explained_variance_ratio = my_pca.get_explained_variance_ratio()
    print(f'Explained variance ratio: {explained_variance_ratio}')
    return my_pca, similar_columns

def y_data(df):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
    y = kmeans.labels_
    return y


def merge_similar_columns(self, df, similar_columns):
    for col1, col2 in similar_columns:
        df[col1] = df[col1] + df[col2].median()  
        df = df.drop(columns=col2)
    return df



def preprocess_and_split(df, y, numerical_columns, categorical_columns, test_size=0.2):  # y should be passed as an argument
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(), categorical_columns)])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_processed = pipeline.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

files = os.listdir()
csv_files = [f for f in files if f.endswith('.csv')]
dataframes = {}
my_kmeans = mykmeans(random_state=0)


for file in csv_files:
    df = process_data(file)
    similar_columns, _ = perform_pca(df)  # Changed here
    df = my_kmeans.merge_similar_columns(df, similar_columns)  # Now similar_columns is available
    y = y_data(df)
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    X_train, X_test, y_train, y_test = preprocess_and_split(df, y, numerical_columns, categorical_columns)
    dataframes[file] = (X_train, X_test, y_train, y_test)


'''
df = pd.read_csv('dataset.csv')
df = df.fillna(value=0)
df = df.drop_duplicates()

columns_to_sum = ['Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 
                  'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 
                  'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 
                  'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 
                  'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
                  'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)']

df['Curricular units'] = df[columns_to_sum].sum(axis=1)
df = df.drop(columns_to_sum, axis=1)


sns.countplot(x='Target', data=df, palette='hls')
plt.show()

kateg_study = df.groupby('Target').mean()

compare_column = 'Target'
columns_to_compare = ['Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance',
                      'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
                      "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
                      'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
                      'Age at enrollment', 'International']

for column in columns_to_compare:
    pd.crosstab(df[column], df[compare_column]).plot(kind='bar')
    plt.title(f'Comparison of {compare_column} for different {column}')
    plt.xlabel(column)
    plt.ylabel(compare_column)
    plt.show()

features = df.columns.drop('Target')
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

for k in features:
    df[k] = (df[k] - df[k].mean())/(df[k].max()-df[k].min())
    
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df.head(5)


type_dict = {'Graduate':0, 'Dropout':1, 'Enrolled':2}
df['Target'] = df['Target'].map(type_dict)

df['Target'].head(5)

X = df[features]
y = df.drop(features, axis=1)['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777, train_size=0.8)
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))

print("y_train_encoded:")
print(y_train_encoded[:5])
print("y_test_encoded:")
print(y_test_encoded[:5])



    



model = Sequential()
model.add(Dense(64, input_dim=23, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train_encoded , epochs=5, batch_size=32)
model.summary()

loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Accuracy: {accuracy * 100}%")



    
    
'''


