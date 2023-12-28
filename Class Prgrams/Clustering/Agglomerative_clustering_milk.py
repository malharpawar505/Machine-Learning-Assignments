import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

milk = pd.read_csv("milk.csv",index_col = 0)

scalar = StandardScaler()
milkscaled = scalar.fit_transform(milk)

linkage_method = "average"
clust = AgglomerativeClustering(n_clusters= 3,linkage= linkage_method)

clust.fit(milkscaled)

print(clust.labels_)

print(silhouette_score(milkscaled, clust.labels_))
