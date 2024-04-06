import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

from eda import process_data, y_data, get_categorical_and_numerical_columns, evaluate_model, preprocess_and_split


def main():
    df = pd.read_csv('/data/raw/dataset.csv')

    df = process_data(df)
    y = y_data(df)
    numerical_columns, categorical_columns = get_categorical_and_numerical_columns(df)
    X_train, X_test, y_train, y_test = preprocess_and_split(df, y, numerical_columns, categorical_columns)

    model1 = DecisionTreeClassifier()
    model2 = RandomForestClassifier(n_estimators=100)  

    ensemble_model = VotingClassifier(estimators=[('dt', model1), ('rf', model2)], voting='soft')
    ensemble_model.fit(X_train, y_train)
    evaluate_model(ensemble_model, X_test, y_test)


    plt.figure(figsize=(10, 6)) 
    for col in numerical_columns:
        plt.subplot(1, len(numerical_columns), numerical_columns.index(col) + 1)
        plt.hist(df[col])
        plt.title(col)
    plt.tight_layout()  
    plt.show()



if __name__ == "__main__":
    main()
