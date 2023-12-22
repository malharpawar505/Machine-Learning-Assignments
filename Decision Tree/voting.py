import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
from sklearn.pipeline import Pipeline

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr,drop_first=True)
x = dum_hr.drop('left',axis = 1)
y = dum_hr['left']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,
                                                 stratify=y,
                                                 random_state=23)

lr = LogisticRegression()
mm_scaler = MinMaxScaler()
knn = KNeighborsClassifier()
pipe_knn = Pipeline([('SCL',mm_scaler),('KNN',knn)])
nb = GaussianNB()

voting = VotingClassifier([('LR',lr),('P_KNN',pipe_knn),('NB',nb)],voting='soft')

voting.fit(x_train,y_train)
y_pred = voting.predict(x_test)
print(accuracy_score(y_test,y_pred))
y_pred_preb= voting .predict_proba(x_test)