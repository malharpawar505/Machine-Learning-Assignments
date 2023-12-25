from lightgbm import LGBMClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import numpy as np

hr = pd.read_csv("satellite.csv",sep=';')
#hr = pd.get_dummies(hr, drop_first=True)
X = hr.drop('classes', axis=1)
y = hr['classes']


l_gbm = LGBMClassifier(random_state=23)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
print(l_gbm.get_params())

params={'n_estimators':[50,75,100],
        'learning_rate':np.linspace(0.001, 0.7, 5),
        'max_depth':[None, 2, 3, 5]}
gcv = GridSearchCV(l_gbm, param_grid=params, cv=kfold,scoring='neg_log_loss')
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

bm_gbm = gcv.best_estimator_
df_imp = pd.DataFrame({'Features':list(X.columns),
                       'Importance':bm_gbm.feature_importances_})
df_imp = df_imp[df_imp['Importance']>0].sort_values('Importance')
plt.barh(df_imp['Features'],df_imp['Importance'])
plt.title("GBM")
plt.show()