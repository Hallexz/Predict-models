from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from analyz import X_train, X_test, y_train, y_test



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
