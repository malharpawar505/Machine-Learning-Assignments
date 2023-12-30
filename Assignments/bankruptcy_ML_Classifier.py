#########################################################b.	Using pipeline, min max scaling + MLP Classifier#######################################################################





import pandas as pd 
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
df = pd.read_csv("Bankruptcy.csv")

X = df.drop('R24', axis=1)
y = df['D'].values

scl_x = MinMaxScaler()
scl_y = MinMaxScaler()

scaled_y = scl_y.fit_transform(y.reshape(-1,1))

mlp = MLPRegressor(random_state=23, 
                    hidden_layer_sizes=(7,5,4,3),
                    activation='relu')
pipe = Pipeline([('SCL', scl_x),('MLP', mlp)])

X_train, X_test, y_train, y_test = train_test_split(X,scaled_y[:,0], 
                               test_size=0.3,
                               random_state=23)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))

############# Grid Search CV #####################
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import KFold
print(pipe.get_params())

kfold = KFold(n_splits=5, shuffle=True, random_state=23)
params = {'MLP__learning_rate':['invscaling','adaptive','constant'],
          'MLP__learning_rate_init': [0.1, 0.2, 0.4],
          'MLP__hidden_layer_sizes':[(10, 7), (7,5,4), (10, 5)]}

gcv_mlp = GridSearchCV(pipe, param_grid=params,verbose=3,
                       cv=kfold)
gcv_mlp.fit(X, scaled_y[:,0])
print(gcv_mlp.best_params_)
print(gcv_mlp.best_score_)
