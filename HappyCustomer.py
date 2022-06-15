import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#loading data
review_data = pd.read_csv(r'ACME-HappinessSurvey2020.csv')

list_of_review = review_data.columns.values.tolist()

X = review_data.drop('Y', axis=1)
y = review_data['Y']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = RandomForestClassifier(n_estimators=30, random_state=0)

classifier.fit(x_train, y_train)

prediction = classifier.predict(x_test)

print(prediction)