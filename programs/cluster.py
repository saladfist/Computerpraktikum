import argparse
import pandas as pd
import time
import os
from functions import * 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="program to find cluster in a dataset of arbitrary dimension"
    )
    parser.add_argument("-d",  "--delta",  default=0.05,type=float    ,help="cube length of the partitioning grid (2*delta), (default: 0.05)")
    parser.add_argument("-eps","--epsilon",default=3,type=float       ,help="epsilon parameter for B set calculations (default: 3)")
    parser.add_argument("-t",  "--tau",  default=2,type=float,help="connection length threshold tau (default: 2)")
    parser.add_argument("-ndata",type=int,default=None,help="Number of data points to read from the dataset (default: all)")
    parser.add_argument("dataset",type=str,help="Name of the dataset (csv file in cluster-data folder), read as <name> or <name>.csv")
    args = parser.parse_args()

    delta = args.delta
    epsilon = args.epsilon
    tau = args.tau
    ndata =args.ndata
    dataset_name=args.dataset
    if dataset_name.endswith(".csv"):
        dataset_name=dataset_name[:-4]
    if ndata is not None:
        df=pd.read_csv((os.getcwd())+f"/cluster-data/{dataset_name}.csv",header=None,nrows=ndata)
    else:
        df=pd.read_csv((os.getcwd())+f"/cluster-data/{dataset_name}.csv",header=None)
    data=df.values.tolist()
    dimension=len(data[0])

    start = time.time()
    remaining_clusters,rho_history,B_history=iteration_over_rho(data,delta,epsilon,tau)
    end = time.time()
    #print outputs
    print("#"*30)
    print("Elapsed Time"+"\n"+f"{end-start:.1f}s")
    print("#"*30)
    print(f"Number of clusters found = {len(remaining_clusters)}")
    print("Cluster  || size")
    for i,cluster in enumerate(remaining_clusters):
        print(f"#{i} \t || {len(cluster)}")
    save_clusters(data,remaining_clusters,dimension,dataset_name)
    save_log(rho_history,B_history,dataset_name)
    if dimension==2 or dimension==3:
        plot_clusters(data,remaining_clusters,dimension,dataset_name)
    