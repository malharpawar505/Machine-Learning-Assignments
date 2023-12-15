# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:08:06 2023

@author: Lenovo
"""

import pandas as pd 
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as  mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression


'''import the following datasets from python and perform train_test_split((80:20),(70:30))'''

df=pd.read_csv("Iris.csv")
X=df[["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]]
y=df["Species"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=30)


#______________________________test_size=0.30________

df=pd.read_csv("Iris.csv")
X=df[["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]]
y=df["Species"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=30)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#____________________________________________________________ 


from sklearn.datasets import load_diabetes
df2=load_diabetes()

X=df2.data
y=df2.target
#print(df2.data)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=45)

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)


#__________________________________________________________ 

from sklearn.datasets import load_diabetes
df2=load_diabetes()

X=df2.data
y=df2.target
#print(df2.data)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=45)

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)


#_____________________________________________

from sklearn.datasets import load_digits
df3=load_digits()

X=df3.data
y=df3.target
#print(df2.data)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=45)

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape) 


#____________________________________ 

from sklearn.datasets import load_digits
df3=load_digits()

X=df3.data
y=df3.target
#print(df2.data)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=45)

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape) 


#___________________________________ 




df4=pd.read_csv("boston.csv")

X=df4[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']]
y=df4['medv']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=23)

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)  



lr=LinearRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

print("R2 values",r2_score(y_test, y_pred))




#_____________________Que 1 ______ 

import matplotlib.pyplot as plt 

#count=100 
np.random.seed(42)

X1=np.random.randint(1,200,100)


#np.random(56)
X2=np.random.randint(1,200,100)
noise=np.random.normal(0,0.2,100)



y=2*X+3*y +noise 

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X1,X2,y,c='r',marker='o')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()


#___________________________2D ______________________ 

import matplotlib.pyplot as plt 

#count=100 
np.random.seed(42)

X1=np.random.randint(1,200,100)


#np.random(56)
X2=np.random.randint(1,200,100)
noise=np.random.normal(0,0.2,100)



y=2*X+3*y +noise 

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X1,X2,y,c='r',marker='o')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()


#___________________________________ 

np.random.seed(42)

X1=np.random.randint(1,200,100)


#np.random(56)
X2=np.random.randint(1,200,100)
noise=np.random.normal(0,0.2,100)

y=2*X1+3*X2 +noise 


df=pd.DataFrame({"Y":y})
df

df['x1']=X1
df['x2']=X2

df

X=df[['x1','x2']]
Y=df['Y']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)

lr=LinearRegression() 
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

print("R2 value is:",r2_score(y_test, y_pred))  

print(lr.coef_)
print(lr.intercept_)



#_____________________________________________ 

df2=pd.read_csv("electricityConsumptionAndProductioction.csv",index_col=0)

df2.describe()

print(np.quantile(df2['Consumption'],[0.25,0.5,0.75]))

print(np.quantile(df2['Production'],[0.25,0.5,0.75]))


#_____________avg electricity production_____________ 

print(np.mean(df2['Nuclear']))


#__________________3rd __________________ 
df2.isnull().sum()


#____4th 

df2.duplicated().sum()


#__________5th ___________ 

plt.figure(figsize = (20,5))
plt.plot(df2.index,df2['Production'],'r')
plt.plot(df2.index,df2['Consumption'],'b')
plt.title('production vs Consumption')
plt.xlabel('Time')
plt.ylabel('Power in MW')
plt.show()

#__________________6 th ________________ 

df_year=df2.groupby('Year').mean()
plt.figure(figsize = (20,5))
plt.plot(df2.index,df2['Production'],'r')
plt.plot(df2.index,df2['Consumption'],'b')

plt.title('year wise consumption vs production')
plt.xlabel('Year')
plt.ylabel('MW')
plt.legend(['Production','Consumption'])
plt.show()

df2=pd.read_csv("electricityConsumptionAndProductioction.csv")
# Reset our index so datetime_utc becomes a column
df2['DateTime' ]=pd. to_datetime(df2['DateTime'])
df2['year']=df2['DateTime'].dt.year
print (df2)
df_year = df2.groupby ("year"). mean()
plt. figure(figsize=(20,5))
plt.plot(df_year.index,df_year['Consumption'], 'r')
plt. plot(df_year. index, df_year['Production'], 'b')
plt.xlabel( 'year')
plt. ylabel ( 'MW')
plt. title('Year wise Consumption vs Production') 
plt. legend (['Consumption', 'Production']) 
plt.grid()
plt. show()


#________________7th ______ 




