import argparse
import pandas as pd
import time
from functions import * 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="program to find cluster in a dataset of arbitrary dimension"
    )
    parser.add_argument("-d",  default=0.05,type=float)
    parser.add_argument("-eps",default=3,type=float)
    parser.add_argument("-t",  default=2.000001,type=float)
    parser.add_argument("dataset",type=str)
    args = parser.parse_args()

    delta = args.d
    epsilon = args.eps
    tau = args.t
    dataset_name=args.dataset
    
    df=pd.read_csv((os.getcwd())+f"/cluster-data/{dataset_name}.csv",header=None,nrows=3000)
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
        plot_clusters(data,remaining_clusters,dimension)
    