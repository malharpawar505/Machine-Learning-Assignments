import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr,drop_first=True)
x = dum_hr.drop('left',axis = 1)
y = dum_hr['left']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size = 0.3,
                                                 stratify = y,
                                                 random_state = 23)

dtc = DecisionTreeClassifier(random_state = 23,max_depth = 2)
dtc.fit(x_train,y_train)

plt.figure(figsize = (10,8))
plot_tree(dtc,feature_names = list(x.columns),
          class_names = ['Working','left'],
          filled = True,fontsize=11)

plt.show()

y_pred = dtc.predict(x_test)
y_pred_prob = dtc.predict_proba(x_test)




##################################### GridsearchCV #####################################
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 23)
params = {'max_depth':[3,4,5,6,7,None]}
dtc = DecisionTreeClassifier(random_state = 23)
gcv = GridSearchCV(dtc, param_grid = params,cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


#####################################Usinng neg_log_loss #######################################




from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 23)
params = {'max_depth':[3,4,5,6,7,None]}
dtc = DecisionTreeClassifier(random_state = 23)
gcv = GridSearchCV(dtc, param_grid = params,cv=kfold,scoring = 'neg_log_loss')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)







