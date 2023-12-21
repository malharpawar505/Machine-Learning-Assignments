import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("diabetes_prediction_dataset.csv",index_col=0)
dum_df = pd.get_dummies(df,drop_first=True)
x=dum_df.drop(["diabetes"],axis=1)
y=dum_df["diabetes"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=23)

from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, log_loss,confusion_matrix,ConfusionMatrixDisplay,classification_report

#Logistic Regression
lr=LogisticRegression()
lr.fit(x_train, y_train)
y_pred=lr.predict(x_test)
print("Accuracy score=",accuracy_score(y_test, y_pred))  
print("Classification Report=",classification_report(y_test, y_pred))

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold


#Logistic using kFold
lr=LogisticRegression()
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
params={'penalty':['l1','l2','elasticnet',None],'l1_ratio':np.linspace(0,1,5)}
gcv=GridSearchCV(lr, param_grid=params,cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_) 
