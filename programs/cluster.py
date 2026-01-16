#!/usr/bin/env python3 
import argparse
import pandas as pd
import time
import os
import glob
from functions import *
from scipy_clustering import *
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="program to find cluster in a dataset of arbitrary dimension"
    )
    parser.add_argument("-d",  "--delta",  default=0.05,type=float    ,help="cube length of the partitioning grid (2*delta), (default: 0.05)")
    parser.add_argument("-eps","--epsilon",default=3,type=float       ,help="epsilon parameter for B set calculations (default: 3)")
    parser.add_argument("-t",  "--tau",  default=2,type=float,help="connection length threshold tau (default: 2)")
    parser.add_argument("-ndata",type=int,default=None,help="Number of data points to read from the dataset (default: all)")
    # parser.add_argument("-dd",type=bool,default=False,help="Determine optimal delta using Delta (default: False)")
    parser.add_argument("-plot",type=bool,default=False,help="Plot the clusters using ggobi-like methods (default: False)")
    parser.add_argument("dataset",type=str,help="Name of the dataset (csv file in cluster-data folder), read as <name> or <name>.csv")
    
    parser.add_argument('--dd', dest="determine_optimal_delta",action='store_true',help="Determine optimal delta using Delta (default: False)")
    parser.set_defaults(determine_optimal_delta=False)
    parser.add_argument("--kmeans",dest="kmeans_clustering",action="store_true",help="Use scipy.cluster as the clustering algorithm (default: False)")
    parser.set_defaults(kmeans_clustering=False)

    args = parser.parse_args()

    
    delta = args.delta
    epsilon = args.epsilon
    tau = args.tau
    ndata =args.ndata
    dataset_name=args.dataset
    if dataset_name.endswith(".csv"):
        dataset_name=dataset_name[:-4]
        
    dataset_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.join("cluster-data", f"{dataset_name}.csv"))
    # for p
    if ndata is not None:
        df=pd.read_csv(dataset_path,header=None,nrows=ndata)
    else:
        df=pd.read_csv(dataset_path,header=None)
    data=df.values.tolist()
    dimension=len(data[0])
    
    standard_clustering=True
    if args.determine_optimal_delta or args.kmeans_clustering:
        standard_clustering=False
    if args.determine_optimal_delta:
        print("#"*30)
        print("Determining optimal delta...")
        start_delta_optimization = time.time()
        delta_opt=determine_optimal_delta(data,epsilon,tau)
        end_delta_optimization = time.time()
        print("Elapsed Time"+"\n"+f"{end_delta_optimization-start_delta_optimization:.1f}s")
        print("#"*30)
        print(f"optimal delta : {delta_opt} ")
        print("#"*30)
        exit()
    
    if args.kmeans_clustering:
        print("#"*30)
        print("Clustering using scipy K-means algorithm...")
        start_scipy=time.time()
        clusters=kmeans(data)
        end_scipy=time.time()
        print("Elapsed Time"+"\n"+f"{end_scipy-start_scipy:.1f}s")
        print(f"Number of clusters found = {len(clusters)}")
        ordered_clusters=sorted(clusters,key=lambda x:len(x),reverse=True)
        for i,cluster in enumerate(ordered_clusters):
            print(f"#{i} \t || {len(cluster)}")
        save_clusters(data,clusters,[],dimension,dataset_name)
        if dimension==2 or dimension==3:
            plot_clusters(data,clusters,[],dimension,dataset_name)
    
        
        
    if standard_clustering:
        start = time.time()
        clusters,unclustered_points,rho_history,B_history=iteration_over_rho(data,delta,epsilon,tau)
        end = time.time()

        print("#"*30)
        print("Elapsed Time"+"\n"+f"{end-start:.1f}s")
        print("#"*30)
        print(f"Number of clusters found = {len(clusters)}")
        print("Cluster  || size")
        ordered_clusters=sorted(clusters,key=lambda x:len(x),reverse=True)
        for i,cluster in enumerate(ordered_clusters):
            print(f"#{i} \t || {len(cluster)}")
        save_clusters(data,clusters,unclustered_points,dimension,dataset_name)
        save_log(rho_history,B_history,end-start,dataset_name)
        if dimension==2 or dimension==3:
            plot_clusters(data,clusters,unclustered_points,dimension,dataset_name,kmeans_used=args.kmeans_clustering)
        