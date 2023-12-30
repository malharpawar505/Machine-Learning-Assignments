'''1.	Consider the dataset in Cases/Bankruptcy/bankruptcy.csv, Taking y=D and X=R1, R2, … R24. Do the following Grid Search with parameters of your choice:
a.	Using pipeline, min max scaling + SDG Classifier
b.	Using pipeline, min max scaling + MLP Classifier
'''
###################################################a.	Using pipeline, min max scaling + SDG Classifier##############################################
import pandas as pd 
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

bank = pd.read_csv("Bankruptcy.csv")
dum_bank = pd.get_dummies(bank, drop_first=True)
X = dum_bank.drop('R2', axis=1)
y = dum_bank['D']

X_train, X_test, y_train, y_test = train_test_split(X,y, 
                               test_size=0.3,
                               stratify=y,
                               random_state=23)

sgd = SGDClassifier(random_state=23,learning_rate='constant',
                    penalty=None, eta0=0.1, loss='log_loss')
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = sgd.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

#####################################################Min Max Scalar ###################################################
mm_scaler = MinMaxScaler()
pipe = Pipeline([('SCL', mm_scaler),('SGD', sgd)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = pipe.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))


############# Grid Search CV #####################
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold
print(pipe.get_params())

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
params = {'SGD__learning_rate':['optimal','adaptive','constant'],
          'SGD__eta0': [0.1, 0.2, 0.4],
          'SGD__penalty':[None, 'l1', 'l2', 'elasticnet']}
gcv_sgd = GridSearchCV(pipe, param_grid=params,verbose=3,
                       cv=kfold, scoring='neg_log_loss')
gcv_sgd.fit(X, y)
print(gcv_sgd.best_params_)
print(gcv_sgd.best_score_)




















