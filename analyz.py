import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten 
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


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


models = {'logistic_regression': LogisticRegression(),
          'svm': SVC(),
          'decision_tree': DecisionTreeClassifier(),
          'random_forest': RandomForestClassifier(),
          'GNB': GaussianNB(),
          'gradient_boosting': GradientBoostingClassifier()}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name}: {accuracy:.2f}') 
    
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))    
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))








    
    




