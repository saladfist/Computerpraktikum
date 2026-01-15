from scipy import cluster
from scipy.spatial.distance import  cdist
import pandas as pd
import numpy as np


def kmeans(data,max_k=12,n_iter=5):
    data_array=pd.DataFrame(data).to_numpy()
    
    #elbow method to determine optimal k for k-means
    K=range(1,max_k)
    inertias=[]
    for k in K:
        best_inertia=float("inf")
        best_centroids=None
        best_labels=None
        for _ in range(n_iter): #multiple initializations for better results    
            centroids,labels=cluster.vq.kmeans(data_array,k)
            inertia=sum(np.min(cdist(data_array,centroids,"euclidean"),axis=1))
            if inertia<best_inertia:
                best_inertia=inertia
                best_centroids=centroids
                best_labels=labels

        inertias.append(best_inertia)

    optimal_k=2
    changes=[]            
    for i in range(1,len(inertias)-1):
        change_normalized=(inertias[i-1]-inertias[i])/(inertias[i-1])
        changes.append(change_normalized)

    if len(changes) > 2:
        # Find the first point where the change is less than half of the maximum change
        max_change = max(changes) 
        for i, change in enumerate(changes):
            if i>=1 and change < 0.5 * max_change: # elbow point
                optimal_k = i + 1  # +1 because K starts at 1
                break
    optimal_centroids,optimal_labels=cluster.vq.kmeans2(data_array,optimal_k)
    clusters=[]
    for i in range(optimal_k):
        cluster_i=set(np.where(optimal_labels==i)[0].tolist())
        clusters.append(cluster_i)
    return clusters