#Generate Random nummber for variable x and use the following equation for y:
#y = 4+3*x+noise
#Apply Linear regresion model and check for MSE
#Apply Ridge Regression and compare the difference between MSE. 
#(use alpha = 1)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_squared_error

#create a synthetic dataset
np.random.seed(42)
x = 2*np.random.rand(100,1)
y = 4+3* x + np.random.randn(100,1)

#split the dataset into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state=42)

#Linear Regression
linear_reg_model = LinearRegression()
linear_reg_model.fit(x_train, y_train)

#make predictions on the test set
y_pred_linear = linear_reg_model.predict(x_test)

#calculate mean Squared Error for linear regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
print("linear Regression - Mean Squared Error:",mse_linear)

#Ridge Regression
ridge_model = Ridge(alpha = 1)#you can experiment with different values of alpha
ridge_model.fit(x_train,y_train)

#make predictions on the test set using Ridge regression
y_pred_ridge = ridge_model.predict(x_test)

#calculate mean Squared  Error for ridge regression
mse_ridge = mean_squared_error(y_test,y_pred_ridge)
print("Ridge Regression - mean Squared Error:",mse_ridge)

#Compare the coefficients of Linear Regression and Ridge Regression
print("\n Linear Regression Cofficients: ",linear_reg_model.coef_)
print("Ridge Regression Coefficients: ",ridge_model.coef_)