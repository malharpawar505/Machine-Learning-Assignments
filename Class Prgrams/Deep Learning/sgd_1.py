import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
from sklearn.preprocessing import MinMaxScalar
from sklearn.pipeline import Pipeline

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr,drop_first=True)
x = dum_hr.drop('left',axis = 1)
y = dum_hr['left']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,stratify= y,random_state = 23)

sgd = SGDClassifier(random_state = 23,learning_rate = 'constant',penalty = None,eta0=0.1)

sgd.fit(x_train,y_train)
y_pred = sgd.predict(x_test)
print(accuracy_score(y_test, y_pred))



######################Min Max Scalar ###############################################################
m_scalar = MinMaxScalar()
pipe = Pipeline([('SCL',mm_scalar),('SGD',sgd)])
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
print(accuracy_score(y_test, y_pred))
      
y_pred_prob = pipe.predict_proba(x_test)
print(log_loss(y_test,y_pred_proba))

#############################Grid seach CV#########################################################
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
print(pipe.get_params())

