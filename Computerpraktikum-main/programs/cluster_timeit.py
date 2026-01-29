#!/usr/bin/env python
import timeit
import subprocess
import argparse


def run_clustering(dataset,d=0.05,eps=3,t=2):
    cmd = ['python', 'cluster.py', dataset, '-d', str(d), '-eps',str(eps), '-t',str(t)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


parser = argparse.ArgumentParser(
        description="program to find cluster in a dataset of arbitrary dimension"
    )
parser.add_argument("-d",  "--delta",  default=0.05,type=float    ,help="cube length of the partitioning grid (2*delta), (default: 0.05)")
parser.add_argument("-eps","--epsilon",default=3,type=float       ,help="epsilon parameter for B set calculations (default: 3)")
parser.add_argument("-t",  "--tau",  default=2,type=float,help="connection length threshold tau (default: 2)")
parser.add_argument("-n",  "--number",  default=3,type=int,help="number of times to run cluster.py (default: 3)")
parser.add_argument("-ndata",type=int,default=None,help="Number of data points to read from the dataset (default: all)")
parser.add_argument("dataset",type=str,help="Name of the dataset (csv file in cluster-data folder), read as <name> or <name>.csv")
args = parser.parse_args()

delta = args.delta
epsilon = args.epsilon
tau = args.tau
number = args.number  
ndata =args.ndata
dataset_name=args.dataset
print(f"Timing cluster.py execution for {dataset_name} ({number} times)... ")
t = timeit.Timer(lambda:run_clustering(dataset_name,delta,epsilon,tau), 
                      setup="from __main__ import run_clustering",
                      )
times=t.timeit(number)
print(f"Average time: {times/number:.4f} seconds")