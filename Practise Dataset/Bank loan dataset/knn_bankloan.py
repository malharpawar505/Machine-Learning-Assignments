


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("bankloan.csv",index_col=0)
x=df.drop(["Personal.Loan","ZIP.Code"],axis=1)
y=df["Personal.Loan"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=23)

from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, log_loss,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier








knn=KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
print("Accuracy score=",accuracy_score(y_test, y_pred))#Accuracy score= 0.908
print("Confusion Metrix=",confusion_matrix(y_test, y_pred))#Confusion Metrix= [[1323   33]
                                                                              #[ 105   39]]
print("Classifiction Report=",classification_report(y_test, y_pred))

#Knn Training
knn=KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred_train=knn.predict(x_train)
print("Accuracy score=",accuracy_score(y_train, y_pred_train)) #Accuracy score= 0.9388571428571428
#kNN using KFold
knn=KNeighborsClassifier()
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
params={'n_neighbors':[1,3,5,7,9,11,13,15,17]}
gcv=GridSearchCV(knn, param_grid=params,cv=kfold,scoring='neg_log_loss')
gcv.fit(x,y)
print(gcv.best_params_) 
print(gcv.best_score_)