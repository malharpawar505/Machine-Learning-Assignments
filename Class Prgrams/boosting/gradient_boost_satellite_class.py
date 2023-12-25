import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier

hr = pd.read_csv("Satellite.csv",sep=';')
#dum_hr = pd.get_dummies(hr, drop_first=True)
X = hr.drop('classes', axis=1)
y = hr['classes']

gbm = GradientBoostingClassifier(random_state=23)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
print(gbm.get_params())
params={'n_estimators':[25,100,150],
        'learning_rate':np.linspace(0.001, 0.8, 10),
        'max_depth':[None, 3,5,7]}
gcv = GridSearchCV(gbm, param_grid=params, cv=kfold, verbose=3,
                   scoring='neg_log_loss')
gcv.fit(X, y)   
print("The Best Parameters: ",gcv.best_params_)
print("The Best Score:",gcv.best_score_)

bm_gbm = gcv.best_estimator_
df_imp = pd.DataFrame({'Features':list(X.columns),
                       'Importance':bm_gbm.feature_importances_})
df_imp = df_imp[df_imp['Importance']>0].sort_values('Importance')
plt.barh(df_imp['Features'],df_imp['Importance'])
plt.title("GBM")
plt.show()