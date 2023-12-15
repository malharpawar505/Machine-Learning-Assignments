import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

boston=pd.read_csv("Boston.csv")
boston

x=boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'black', 'lstat']]
y=boston['medv']
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)