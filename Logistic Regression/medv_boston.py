#median value of ownder occupied homes

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

boston = pd.read_csv("Boston.csv")
x = boston.drop('medv',axis = 1).values
y = boston['medv'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=23)
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print(r2_score(y_test, y_pred))



#######################using pipeline####################################################
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Read the data
boston = pd.read_csv("Boston.csv")

# Separate features and target
X = boston.drop('medv', axis=1)
y = boston['medv']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Define the column transformer
# In this case, we'll use StandardScaler for all features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)
    ]
)

# Create a pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=3))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("R-squared Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
