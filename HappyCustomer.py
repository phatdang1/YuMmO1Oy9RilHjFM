from result_report import reportAndScore
from data_processing import readAndProcessCsv, dropColumn

model = "RFC"
#loading data
review_data = readAndProcessCsv('ACME-HappinessSurvey2020.csv', False)

list_of_review = review_data.columns.values.tolist()

#remove unimportant features
review_data = review_data.drop(['X4', 'X6', 'X2'], axis=1 )

# seperate result from training data
X, y = dropColumn(review_data, 'Y')

# print out the result and score
reportAndScore(X, y, model, 0.5)
