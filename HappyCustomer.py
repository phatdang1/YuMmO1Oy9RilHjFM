import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#loading data
review_data = pd.read_csv(r'ACME-HappinessSurvey2020.csv')

list_of_review = review_data.columns.values.tolist()

review_data = review_data.drop(['X4', 'X6'], axis=1 )
X = review_data.drop(['Y'], axis=1)
y = review_data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = RandomForestClassifier(n_estimators=10, random_state=0)

classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)

print(prediction)
print(classification_report(y_test, prediction))
print(accuracy_score(y_test, prediction))

important_feats = pd.Series(classifier.feature_importances_, index = X.columns)
print(important_feats.nlargest(10))