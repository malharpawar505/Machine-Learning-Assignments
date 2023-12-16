#Generate a random dataset using 'make_classification'function and apply
#stratifies k-fold cross validation technique . show the accuracy for each fold. 


#import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import kfold
from sklear.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import stratified_kf
from sklearn.metrics import accuracy_score
import pandas as pd

#Generte a synthetic dataset for binary classification
x,y = make_classification(
    n_samples = 1000,
    n_features = 10,
    n_informative = 8,
    n_redundant = 2,
    n_clusters_per_class = 2,
    flip_y=0.1,
    random_state = 42
)

#create a Dataframe for visualization purposes
df = pd.DataFrame(x,columns=[f'feature_{i}' for i in range(x.shape[1])])
df['target'] = y

#Display the first few rows of the dataset
print(df.head())

#create a logistic regression model
model = LogisticRegression()

#specify the number of folds
k_folds = 5

#create a stratifiesFold object
stratifies_kf = StratifiedKFold(n_splits=k_folds,shuffle = True,random_state=42)

#perform stratified k-fold cross validation
accuracy_scores=[]
for train_index,test_index in stratified_kf.spilt(x,y):
    x_train,x_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
    
    #train the model
    model.fit(x_train,y_train)
    
    #make predictions on the test set
    y_pred = model.predict(x_test)
    
    #calculate accuracy and store it
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_scores.append(accuracy)
    
    #Display the accuracy for each fold
for i,accuracy in enumerate(accuracy_scores):
    print(f'Fold{i+1}Accuracy:{accuracy:.2f}')
        
    #Display the Average accuracy across all folds
    print(f'Average Accuracy:{np.mean(accuracy_scores):.2f}')
    
    
    
    




