import pandas as pd 
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold 
sat = pd.read_csv("Satellite.csv",sep=';')
X = sat.drop('classes', axis=1)
y = sat['classes']



X_train, X_test, y_train, y_test = train_test_split(X,y, 
                               test_size=0.15,
                               stratify=y,
                               random_state=23)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
## with default params
params ={}
dtc = DecisionTreeClassifier(random_state=23)
gcv_default = GridSearchCV(dtc, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv_default.fit(X_train, y_train)
pd_cv = pd.DataFrame( gcv_default.cv_results_ )
print(gcv_default.best_params_)
print(gcv_default.best_score_)

dtc.fit(X_train, y_train)
y_pred_prob = dtc.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))