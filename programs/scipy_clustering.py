from scipy import cluster
from scipy.spatial.distance import  cdist
import pandas as pd
import numpy as np


def kmeans(data):
    data_array=pd.DataFrame(data).to_numpy()
    
    #elbow method to determine optimal k for k-means
    K=range(1,6)
    distortions=[]
    inertias=[]
    for k in K:
        centroids,labels=cluster.vq.kmeans2(data_array,k)
        distortions.append(sum(np.min(cdist(data_array,centroids,"euclidean"),axis=1)**2)/data_array.shape[0])
        inertias.append(sum(cdist(data_array,centroids,"euclidean")**2))
    diff=np.diff(np.diff(distortions,prepend=distortions[0],append=distortions[-1]),prepend=distortions[0],append=distortions[-1])
    optimal_k=int(np.argmax(diff))
    
    optimal_centroids,optimal_labels=cluster.vq.kmeans2(data_array,optimal_k)
    clusters=[]
    for i in range(optimal_k):
        cluster_i=set(np.where(optimal_labels==i)[0].tolist())
        clusters.append(cluster_i)
    return clusters