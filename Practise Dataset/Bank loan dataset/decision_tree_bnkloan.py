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

#Decision Tree
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

dtc=DecisionTreeClassifier(random_state=23,max_depth=2)
dtc.fit(x_train, y_train)
plt.figure(figsize=(10,8))
plot_tree(dtc,feature_names=list(x.columns),class_names=['0','1'],filled=True,fontsize=11)
plt.show()

y_pred=dtc.predict(x_test)
y_pred_prob=dtc.predict_proba(x_test)

print("Acuracy score=",accuracy_score(y_test, y_pred)) #Acuracy score= 0.9653333333333334
print("Classification Report=",classification_report(y_test, y_pred))
