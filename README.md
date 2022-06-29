# 3nxnFeQLJQzPbZY3
Machine learning project - Happy Customer

Data Description:

feedbacks from customers based on on-demand delivery to our customers. During the COVID-19 pandemic.

The Data below already encoded into digital number based on custormer scores based on the 6 categories below:

Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers
X1 = my order was delivered on time
X2 = contents of my order was as I expected
X3 = I ordered everything I wanted to order
X4 = I paid a good price for my order
X5 = I am satisfied with my courier
X6 = the app makes ordering easy for me

Attributes X1 to X6 indicate the responses for each question and have values from 1 to 5 where the smaller number indicates less and the higher number indicates more towards the answer.

Download Data:

https://drive.google.com/open?id=1KWE3J0uU_sFIJnZ74Id3FDBcejELI7FD

Goal(s):

Predict if a customer is happy or not based on the answers they give to questions asked.

Attempts:
In this project I use the following model: K-Nearest Neighbor, and Random Forest Classifier.
Using K-Nearest Neighbor:

              precision    recall  f1-score   support

           0       0.67      0.62      0.64        13
           1       0.64      0.69      0.67        13

    accuracy                           0.65        26
   macro avg       0.65      0.65      0.65        26
weighted avg       0.65      0.65      0.65        26

Using Random Forest Classifier:

              precision    recall  f1-score   support

           0       0.86      0.58      0.69        33
           1       0.66      0.90      0.76        30

    accuracy                           0.73        63
   macro avg       0.76      0.74      0.73        63
weighted avg       0.77      0.73      0.72        63

Result:
Random Forest Classification and Neural Network give the highest accuracy, percision and f1-score. 

Discussion:
Due to the small number of training data, the highest accuracy that this model can reach is 73%. 


