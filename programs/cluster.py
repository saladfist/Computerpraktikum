#!/usr/bin/env python3 
import argparse
import pandas as pd
import time
import os
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
    
    parser.add_argument('--v', dest="verbose_output",action='store_true',help="Verbose output (default: False)")
    parser.set_defaults(verbose_output=False)
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
    dataset_basename = os.path.basename(dataset_name)   #remove path if cluster data is given with path

    if dataset_basename.endswith(".csv"):
        dataset_name = dataset_basename[:-4]  
    else:
        dataset_name = dataset_basename
        
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
        print("#"*50)
        print("Determining optimal delta...")
        start_delta_optimization = time.time()
        delta_opt=determine_optimal_delta(data,epsilon,tau)
        end_delta_optimization = time.time()
        print("Elapsed Time"+"\n"+f"{end_delta_optimization-start_delta_optimization:.3f}s")
        print("#"*50)
        print(f"optimal delta : {delta_opt} ")
        print("#"*50)
        exit()
    
    if args.kmeans_clustering:
        print("#"*50)
        print("Clustering using scipy K-means algorithm...")
        start_scipy=time.time()
        data_dict=kmeans(data)
        
        end_scipy=time.time()
        clusters=df[df["cluster"]!=0].groupby("cluster")["idx"].apply(list).tolist()
        df=pd.DataFrame(data_dict.values())
        print("Elapsed Time"+"\n"+f"{end_scipy-start_scipy:.3f}s")
        print(f"Number of clusters found = {len(clusters)}")
        ordered_clusters=sorted(clusters,key=lambda x:len(x),reverse=True)
        for i,cluster in enumerate(ordered_clusters):
            print(f"#{i} \t || {len(cluster)}")
        print("#"*50)
        save_started=time.time()
        save_clusters(df,dimension,dataset_name)
        save_ended=time.time()
        if args.verbose_output:
            print("Elapsed Time for saving results"+"\n"+f"{save_ended-save_started:.3f}s")
            print("#"*50)
        if dimension==2 or dimension==3:
            plot_clusters(df,dimension,dataset_name)
    
        
        
    if standard_clustering:
        start = time.time()
        data_dict,rho_history,B_history=backwards_rho_iteration(data,delta,epsilon,tau)
        end = time.time()

        df=pd.DataFrame(data_dict.values())
        clusters=df[df["cluster"]!=0].groupby("cluster")["idx"].apply(list).tolist()

        print("#"*50)
        print("Elapsed Time"+"\n"+f"{end-start:.3f}s")
        print("#"*50)
        print(f"Number of clusters found = {len(clusters)}")
        print("Cluster  || size")
        ordered_clusters_by_length=sorted(clusters,key=lambda x:len(x),reverse=True)
        for i,cluster in enumerate(ordered_clusters_by_length):
            print(f"#{i} \t || {len(cluster)}")
        print("#"*50)
        
        save_started=time.time()
        save_clusters(df,dimension,dataset_name)
        save_ended=time.time()
        if args.verbose_output:
            print("Elapsed Time for saving results"+"\n"+f"{save_ended-save_started:.3f}s")
            print("#"*50)
        save_log_started=time.time()
        save_log(rho_history,B_history,end-start,dataset_name)
        save_log_ended=time.time()
        if args.verbose_output:
            print("Elapsed Time for saving logs"+"\n"+f"{save_log_ended-save_log_started:.3f}s")
            print("#"*50)
        if dimension==2 or dimension==3:
            plot_started=time.time()
            plot_clusters(df,dimension,dataset_name,kmeans_used=args.kmeans_clustering)
            plot_ended=time.time()
            if args.verbose_output:
                print("Elapsed Time for plotting and saving plots"+"\n"+f"{plot_ended-plot_started:.3f}s")
                print("#"*50)
        