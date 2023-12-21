import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("bankloan.csv",index_col=0)
x=df.drop(["Personal.Loan","ZIP.Code"],axis=1)
y=df["Personal.Loan"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=23)

from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, log_loss,confusion_matrix,ConfusionMatrixDisplay,classification_report

#Navive bayes
from sklearn.naive_bayes import BernoulliNB
nb=BernoulliNB()
nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)
print("Accuracy score=",accuracy_score(y_test, y_pred)) #Accuracy score= 0.9033333333333333
print("Classification Report=",classification_report(y_test, y_pred))

#Navie bayes using KFold
nb=BernoulliNB()
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
params={'alpha':np.linspace(0.001,2,20)}
gcv=GridSearchCV(nb, param_grid=params,cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_) 
print(gcv.best_score_)