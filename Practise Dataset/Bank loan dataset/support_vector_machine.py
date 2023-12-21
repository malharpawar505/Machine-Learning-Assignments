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


#Train SVM model
from sklearn.svm import SVC

svm=SVC(kernel='linear')
svm.fit(x_train, y_train)

#make prediction on test set
y_pred=svm.predict(x_test)
#Evaluate the model

print("Acuracy score=",accuracy_score(y_test, y_pred)) #Acuracy score= 0.9533333333333334
print("Confusion Metrix=",confusion_matrix(y_test, y_pred))  #Confusion Metrix= [[1343   13]
                                                                               # [  57   87]]
                                                                               
print("Classifiction Report=",classification_report(y_test, y_pred))

#SVM Train
svm=SVC(kernel='linear')
svm.fit(x_train, y_train)

#make prediction on test set
y_pred=svm.predict(x_train)
#Evaluate the model

print("Acuracy score=",accuracy_score(y_train, y_pred))