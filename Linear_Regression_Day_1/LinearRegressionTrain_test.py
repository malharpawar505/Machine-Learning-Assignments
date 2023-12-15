
import numpy as np 
import pandas as pd 
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

df=pd.read_csv("Insure_auto.csv",index_col=0)

#train=df.iloc[[1,4,5,7,2,9,3],:]
#test=df.iloc[[6,0,8],:]
X=df[['Home','Automobile']]
y=df['cost']

train,test=train_test_split(df,test_size=0.3,random_state=23)
X_train=train[['Home','Automobile']]
y_train=train['Operating_Cost']

X_test=test[['Home','Automobile']]
y_test=test['Operating_Cost']

print(X_train.shape)
print(X_test.shape)
print(y_test.shape) 
print(y_train.shape)
print(X_train)
print(y_train)
print(X_test)
lr=LinearRegression() 
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)
print("R2 Score is ",r2_score(y_test, y_pred))
