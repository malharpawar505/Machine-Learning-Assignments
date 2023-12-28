import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

arrest = pd.read_csv("USArrests.csv",index_col = 0)

scalar = StandardScaler()
milkscaled = scalar.fit_transform(arrest)

ks = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
scores=[]
for i in ks:
    clust = KMeans(n_clusters = i)

    clust.fit(milkscaled)

    scores.append(clust.inertia_)
    
plt.scatter(ks,scores,c = 'red')
plt.plot(ks,scores)
plt.xlabel("No.of Clusters")
plt.ylabel("WSS")
plt.title("Scree plot")
plt.show()
