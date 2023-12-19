import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr,drop_first = True)
x = dum_hr.drop('left',axis = 1).values
y = dum_hr['left'].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,stratify=y,random_state = 23)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred =knn.predict(x_test)
y_pred_prob=knn.predict_proba(x_test)
print(accuracy_score(y_test,y_pred))
print(log_loss(y_test,y_pred_prob))


##########################grid-search###############################
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

knn = KNeighborsClassifier()
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state = 23)

print(knn.get_params())

params = {'n_neighbors':[1,3,5,7,9,11,13,15,17]}

gcv = GridSearchCV(knn, param_grid = params,cv=kfold,scoring = 'neg_log_loss')

gcv.fit(x,y)

print(gcv.best_params)
print(gcv.best_score)
