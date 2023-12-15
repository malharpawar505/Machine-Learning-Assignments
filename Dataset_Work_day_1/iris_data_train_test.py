
#70 % train , 30 Test on iris 
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = pd.read_csv("iris.csv")

#function is a utility in machine learning that is used to split a dataset into training and testing sets
iris_train, iris_test = train_test_split(iris, test_size = 0.3)

x_train = iris_train[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']]
y_train = iris_train['Species']

x_test = iris_test[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']]
y_test = iris_test['Species']

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

#OUTPUT 
#(105, 4) (105,)
#(45, 4) (45,)