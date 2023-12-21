import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

bnkloan = pd.read_csv("bankloan.csv",index_col=0)


#converting categorical data into numeric 
dum_bnkloan = pd.get_dummies(bnkloan,drop_first=True)
print(bnkloan['Personal.Loan'].value_counts())
print(bnkloan['Personal.Loan'].value_counts(normalize = True)*100)

#Spliting data into training and testing set 70% training and 30% testing
train,test = train_test_split(dum_bnkloan,test_size=0.3,stratify=bnkloan['Personal.Loan'],random_state=23)

# Separate features (x) and target variable (y) for both training and testing sets.
x_train = train.drop('Personal.Loan','ZIP.code',axis=1)
y_train = train['Personal.Loan']
x_test = test.drop('Personal.Loan',axis=1)
y_test = test['Personal.Loan']

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