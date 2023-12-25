import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr , drop_first=True)
x = dum_hr.drop('left',axis = 1)
y = dum_hr['left']

gbm = GradientBoostingClassifier(random_state = 23)
kfold = StratifiedKFold(n_splits=5,shuffle = True , random_state = 23)
print(gbm.get_params())

params = {'n_estimators':[50,75,100],
          'learning_rate':np.linspace(0.001, 0.7,5),
          'max_depth':[None,2,3,5]}
gcv = GridSearchCV(gbm, param_grid=params,cv = kfold,verbose = 3,scoring= 'neg_log_loss')

gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)




