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
                               test_size=0.3,
                               stratify=y,
                               random_state=23)
dtc = DecisionTreeClassifier(random_state=23, max_depth=2)
dtc.fit(X_train, y_train)

plt.figure(figsize=(10,8))
plot_tree(dtc,feature_names=list(X.columns),

               filled=True, fontsize=11) 
plt.show()

y_pred = dtc.predict(X_test)
y_pred_prob = dtc.predict_proba(X_test)


### Training set accuracy
y_pred_trn = dtc.predict(X_train)
print(accuracy_score(y_train, y_pred_trn))

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
params = {'min_samples_split':[2,3,4,5,6,10],
          'max_depth': [2,3,4,5,6,7],
          'min_samples_leaf':[1,2,3,4,5,6,10]}
dtc = DecisionTreeClassifier(random_state=23)
gcv = GridSearchCV(dtc, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X_train, y_train)
print(gcv.best_params_)
print(gcv.best_score_)
bm = gcv.best_estimator_
y_pred_prob = bm.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))
#Output
#0.6194493783303731
#{'max_depth': 5, 'min_samples_leaf': 10, 'min_samples_split': 2}
#-0.6618775098630236
#0.667811406547557

