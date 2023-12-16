#trained the whole models on the basis of HR Dataset
#whether employee leave the company or not . 


import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

hr = pd.read_csv("HR_comma_sep.csv")

#converting categorical data into numeric 
dum_hr = pd.get_dummies(hr,drop_first=True)
print(hr['left'].value_counts())
print(hr['left'].value_counts(normalize = True)*100)

#Spliting data into training and testing set 70% training and 30% testing
train,test = train_test_split(dum_hr,test_size=0.3,stratify=hr['left'],random_state=23)

# Separate features (x) and target variable (y) for both training and testing sets.
x_train = train.drop('left',axis=1)
y_train = train['left']
x_test = test.drop('left',axis=1)
y_test = test['left']

print(y_train.value_counts(normalize=True)*100)
print(y_test.value_counts(normalize = True)*100)

#building logistic regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.coef_)
print(lr.intercept_)

# Use the trained model to make predictions on the testing set.
y_pred = lr.predict(x_test)
y_pred_prob = lr.predict_proba(x_test)

#Evaluate the accuracy of the model using the accuracy_score function. 
print(accuracy_score(y_test,y_pred))