# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:23:00 2023

@author: Lenovo
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
#from sklearn.model_selection import Ridge
from sklearn.linear_model import Lasso



df5=pd.read_csv("salaries.csv")

dum_d2=pd.get_dummies(df5,drop_first=True)

    
X=dum_d2.drop('salary',axis=1)
y=dum_d2['salary']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=23)

alphas=np.linspace(0.001,15,20)
scores=[]
for v in alphas:
    ls=Lasso(alpha=v)
    ls.fit(X_train,y_train)

    y_pred=ls.predict(X_test)
    scr=r2_score(y_test, y_pred)
    scores.append(scr)
    print("Alpha=",v,"R2=",scr)
i_max=np.argmax(scores)
print("Best alphas:",alphas[i_max])
print("Best scores:",scores[i_max])


plt.plot(alphas,scores)
plt.scatter(alphas,scores,c='r')
plt.xlabel('alpha')
plt.ylabel('Scores')
plt.title("lasso  Regression plot")  
plt.show()
