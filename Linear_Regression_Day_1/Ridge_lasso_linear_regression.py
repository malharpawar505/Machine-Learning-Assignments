# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:58:37 2023

@author: Lenovo
"""



#______________________________diamands -try linear ,lasso ,ridged regression _________________


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
from sklearn.linear_model import Ridge



df3=pd.read_csv("Diamonds.csv")

dum_d=pd.get_dummies(df3)

    
X=dum_d.drop('price',axis=1)
y=dum_d['price']

#dia_train,dia_test=train_test_split(dum_d,test_size=0.3,random_state=23)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=23)



print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)  



lr=LinearRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

print("R2 values",r2_score(y_test, y_pred)) 


#________________ridged 


df3=pd.read_csv("Diamonds.csv")

dum_d=pd.get_dummies(df3)

    
X=dum_d.drop('price',axis=1)
y=dum_d['price']

#dia_train,dia_test=train_test_split(dum_d,test_size=0.3,random_state=23)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=23)


alphas=np.linspace(0.001,15,20)
scores=[]
for v in alphas:
    rid=Ridge(alpha=v)
    rid.fit(X_train,y_train)

    y_pred=rid.predict(X_test)
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
plt.title("Ridge Regression plot")  
plt.show()


#_______________________________lasso 

df3=pd.read_csv("Diamonds.csv")

dum_d=pd.get_dummies(df3)

    
X=dum_d.drop('price',axis=1)
y=dum_d['price']

#dia_train,dia_test=train_test_split(dum_d,test_size=0.3,random_state=23)

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
plt.title("Lasso Regression plot")  
plt.show()
