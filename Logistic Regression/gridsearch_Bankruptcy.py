from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

bn =pd.read_csv("Bankruptcy.csv",index_col = 0)


lr = LogisticRegression()

x = bn.drop(['D','YR'],axis=1)
y = bn['D']

kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=23)
params = {'penalty':['l1','l2','elasticnet',None],'l1_ratio':np.linspace(0,1,5),
          'solver':['lbfgs','liblinear',
                    'newton-cg','newton-choleshy','saga']}

gcv = GridSearchCV(lr, param_grid = params,cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame(gcv.cv_results_)


######################Inferecing : Predicting on unlabelled data###############################
bm = LogisticRegression(l1_ratio=0.0,penalty = 'l1',solver='liblinear')

bm.fit(x,y)

########unlabelled dataset
tst = pd.read_csv("testBankruptcy.csv",index_col = 0)

##########Inferencing : Predicting on unlabelled data
predictions = bm.predict(tst)
print(predictions)


# OR do with direct gcv.best_estimator_
b_model = gcv.best_estimator_
predictions = b_model.predict(tst)
print(predictions)

