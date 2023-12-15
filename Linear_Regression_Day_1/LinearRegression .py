# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:10:06 2023

@author: Lenovo
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


x=np.array([3,5,6,1,7,8,9,10])
y=3*x+5 

np.corrcoef(x,y)

lr=LinearRegression()
lr.fit(x.reshape(-1,1),y)

print(lr.intercept_)
print(lr.coef_)

plt.scatter(x,y)
plt.plot(x,y,c='red')
plt.show() 
#_________________________________________


x=np.array([3,5,6,1,7,8,9,10])
r=np.array([0.1,-0.2,0.3,0.4,0.9,1.2,-1.3,1])
y=3*x+5 
x=x+r
np.corrcoef(x,y)

lr=LinearRegression()
lr.fit(x.reshape(-1,1),y)

print(lr.intercept_)
print(lr.coef_)

ycap=2.76805098*x+5.590272428041182
plt.scatter(x,y)
plt.plot(x,ycap,c='red')
plt.show() 


#____________________________ 


df=pd.read_csv("pizza.csv")

x=df['Promote'].values
y=df['Sales'].values

np.corrcoef(x,y)

lr=LinearRegression()
lr.fit(x.reshape(-1,1),y)

print(lr.intercept_)
print(lr.coef_)

vals=np.array([75,50,40])
y_pred=lr.predict(vals.reshape(-1,1))
print(y_pred)
#----


ycap=23.50640302*x+5.4858653632529695
plt.scatter(x,y)
plt.plot(x,ycap,c='red')
plt.show() 


ycap1=23.50640302*75+5.4858653632529695
print("at x=75",ycap1)
ycap2=23.50640302*50+5.4858653632529695
print("at x=50",ycap2)
ycap3=23.50640302*40+5.4858653632529695
print("at x=40",ycap3)





#another way of passing values 

a=np.array([75,50,40])
ycap=23.50640302*a+5.4858653632529695
print(ycap)


#_____________________________________________insure_auto _______________ 


df=pd.read_csv("insure_auto.csv",index_col=0)

y=df['Operating_Cost'].values
x=df[['Home','Automobile']].values

#np.corrcoef(x,y)

lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)


#a=[100,500]
x1=100
x2=500

y_inter1=167.32668857*x1+54.10529229*x2-10084.213130948774
#print(y_inter1)

#y_inter=167.32668857*100+54.10529229*500-10084.213130948774
print(y_inter1)

#plot_axes.scatter3D(y, x, z1)



#__________________________________________predict function_________________ 

df=pd.read_csv("insure_auto.csv",index_col=0)

y=df['Operating_Cost'].values
x=df[['Home','Automobile']].values

#np.corrcoef(x,y)

lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)

vals=np.array([[100,500],
               [200,700],
               [500,200]])
y_pred=lr.predict(vals)
y_pred 

#______________________________

boston=pd.read_csv("Boston.csv")
x=boston['crim'].values
y=boston['medv'].values

lr=LinearRegression()
lr.fit(x.reshape(-1,1),y)

print(lr.intercept_)
print(lr.coef_)

print(boston['crim'].corr(boston['medv']))

ycap=lr.predict(x.reshape(-1,1))

mean_sqe_error=np.square(y-ycap).mean(axis=0)
mean_sqe_error


mean_abs_values=np.absolute(y-ycap).mean(axis=0)
mean_abs_values

#----------------using sklearn.metrics ----------- 

print(MAE(y,ycap))
print(MSE(y,ycap))


#_____________________________________

df=pd.read_csv("insure_auto.csv",index_col=0)

y=df['Operating_Cost'].values
x1=df['Home'].values
x2=df['Automobile'].values


#np.corrcoef(x,y)

lr=LinearRegression()
lr.fit(x1.reshape(-1,1),y)

ycap=lr.predict(x1.reshape(-1,1))

print("MSE for x1", MSE(y,ycap))
print("MAE for x1",MAE(y,ycap))

#__________________________________________ 

lr2=LinearRegression()
lr2.fit(x2.reshape(-1,1),y)

ycap=lr.predict(x2.reshape(-1,1))

print("MSE for x2", MSE(y,ycap))
print("MAE for x2",MAE(y,ycap))




#___________________

df=pd.read_csv("insure_auto.csv",index_col=0)

y=df['Operating_Cost']
X=df[['Home','Automobile']]

lr2=LinearRegression()
lr2.fit(X,y)

ycap=lr2.predict(X)

print("MSE for X", MSE(y,ycap))
print("MAE for X",MAE(y,ycap))


#_________________________coefficient of Determination_____________________ 


boston=pd.read_csv("Boston.csv")
x=boston['crim'].values
y=boston['medv'].values

lr=LinearRegression()
lr.fit(x.reshape(-1,1),y)

print(lr.intercept_)
print(lr.coef_)

print(boston['crim'].corr(boston['medv']))

ycap=lr.predict(x.reshape(-1,1))

print(1-(np.sum((y-ycap)**2))/np.sum((y-y.mean())**2))

print(r2_score(y,ycap))


#______________________ 


df4=pd.read_csv("boston.csv")

X=df4[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']]
y=df4['medv']

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=45)

#print(X_train.shape)
#print(X_test.shape)
#print(y_test.shape)
#print(y_train.shape)  



lr=LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

#print("R2 values",r2_score(X,y))
