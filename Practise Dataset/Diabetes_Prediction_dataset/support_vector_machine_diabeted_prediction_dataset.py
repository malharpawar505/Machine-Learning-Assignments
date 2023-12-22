import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#read csv file
df=pd.read_csv("diabetes_prediction_dataset.csv",index_col=0)

dum_df = pd.get_dummies(df,drop_first=True)
x=dum_df.drop(["diabetes"],axis=1)
y=dum_df["diabetes"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=23)

from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, log_loss,confusion_matrix,ConfusionMatrixDisplay,classification_report


from sklearn.svm import SVC

svm=SVC(kernel='linear')
svm.fit(x_train, y_train)

#make prediction on test set
y_pred=svm.predict(x_test)
#Evaluate the model

print("Acuracy score=",accuracy_score(y_test, y_pred)) 
print("Confusion Metrix=",confusion_matrix(y_test, y_pred))                                                                                 # [  57   87]]
                                                                               
print("Classifiction Report=",classification_report(y_test, y_pred))

print()