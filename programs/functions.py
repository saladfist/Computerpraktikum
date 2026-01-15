import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import os 
from collections import deque, defaultdict


def get_cubes_dict(data,cube_length,dimension):
    '''
    data: lsit of data points
    cube_length: float 2delta
    dimension: float dimension of data
    returns:
    #cubes_dict: a dictionary with cube indices as keys and counts of datapoints in each cube as values
    #cubes_point_map: a dictionary with each cube containing points with all data point indices in that cube
    '''
    cubes_dict={} #dictionary to hold the cubes belonging to data points and their counts 
    cubes_point_map=defaultdict(list)
    cubes_coords_map={}
    
    for point in range(len(data)):
        cube_idx=tuple([int((data[point][d]+1)/(cube_length)) for d in range(dimension)])
        if cube_idx not in cubes_dict:
            cubes_dict[cube_idx]=0
        cubes_dict[cube_idx]+=1 # if data in cube +=1
        cubes_point_map[cube_idx].append(point)
    for cube,points in cubes_point_map.items():
        #get as entry to the cube a tuple which contains 2*dimension values which correspond to the min and max coordinate of points in each dimension
        min_coords=[min(data[p][d] for p in points) for d in range(dimension)]
        max_coords=[max(data[p][d] for p in points) for d in range(dimension)]
        cubes_coords_map[cube]=(tuple(min_coords),tuple(max_coords))
    #we only need the points which are the closest to the cube edges

    return cubes_dict, cubes_coords_map
# iteration over thresholds and find M intervals
def get_M(rho,h_dict): #calculates sets Mρ := {x : hD,δ (x) ≥ ρ}
    #M: list of cube indices (as tuples)
    M=[]
    for i,h in h_dict.items():
        # print("i,h",i,h)
        if h>=rho:
            M.append(i)
    return M


def tau_connected_clusters(data,M,tau,delta,dimension,cubes_dict,cubes_coords_map):
    """
    M list of cube indices (as tuples)
    tau float
    delta float
    dimension int
    Returns: list of clusters, where each cluster is a set of cube indices
    """

    # cluster cubes by connectivity of their centers: 
    # center-distance <= tau -> connected 
    # center-distance <= tau+2delta -> check wheter these cubes contain points close enough to be connected 

    def cubes_connected(c1,c2):
        dist=[c1[d]-c2[d] for d in range(dimension)]
        dist_abs=max(abs(_) for _ in dist)
        if dist_abs<=tau/(2*delta):
            return True
        if dist_abs>tau/(2*delta)+1:
            return False
        # check whether there exist points x in c1 and y in c2 with distance <=
        
        c1_min,c1_max=cubes_coords_map[c1]
        c2_min,c2_max=cubes_coords_map[c2]
        
        for d in range(dimension):
            if c1_min[d]>c2_max[d]+tau or c2_min[d]>c1_max[d]+tau:
                return False 
        return True
        
        
    
    visited = set()
    clusters = []
    
    for start_cube in M:
        if start_cube in visited:
            continue
        stack = [start_cube]
        cluster = set()
        
        while stack:
            c = stack.pop()
            if c in visited:
                continue
            visited.add(c)
            cluster.add(c)
            
            for other in M:
                if other not in visited and cubes_connected(c,other):
                    stack.append(other)

        clusters.append(cluster)

    # order clusters by size (largest first).
    clusters.sort(key=len, reverse=True)
    #get the number of datapoints in the largest and second largest cluster
    #cubes_dict contains counts of datapoints per cube
    B1 = sum(cubes_dict[c] for c in clusters[0]) if len(clusters) > 0 else 0
    B2 = sum(cubes_dict[c] for c in clusters[1]) if len(clusters) > 1 else 0
    return clusters, B1, B2


def drop_clusters(B,h_dict,rho,epsilon): #drop B if it contains no h with h geq ρ + 2ε; B contains cube indices
    remaining_clusters=[]
    for cluster in B:
        # For each cube in cluster, find max h-value among points in that cube
        max_h = 0
        for point_idx, h in h_dict.items():
            if point_idx in cluster:
                max_h = max(max_h, h)
        if max_h >= rho + epsilon:
            remaining_clusters.append(cluster)
    return remaining_clusters

def iteration_over_rho(data,delta,epsilon_factor,tau_factor):
    
    dimension=len(data[0])
    cubes_dict,cubes_point_map = get_cubes_dict(data, 2*delta, dimension)

    #calulate h_values for all data points
    n=len(data)
    h_dict={}
    for x in data:
        cube_idx_of_x = tuple([int(((x_d)+1)/(2*delta)) for x_d in x])
        h_dict[cube_idx_of_x]=cubes_dict.get(cube_idx_of_x, 0)/(n*2**dimension*delta**dimension)
    h_max = max(h_dict.values())
    
    epsilon=epsilon_factor*(h_max/(n*(2*delta)**dimension))**.5
    tau=tau_factor*delta
    rho_step=1/(n*(2*delta)**dimension)

    rho_history=[]
    B_history=[]
    # find equivalence classes and clusters
    rho=0
    
    while True:
        M=get_M(rho,h_dict)
        B_current,B1,B2=tau_connected_clusters(data,M,tau,delta,dimension,cubes_dict,cubes_point_map)
        remaining_clusters=drop_clusters(B_current,h_dict,rho,epsilon)
        
        if len(remaining_clusters)!=1: #or (M_initial==1 and multiple_clusters): # I think if there exist only 1 cluster B and we dont stop the recursion, then we just increase rho until no clusters remain which leads to large gaps in the clusters, might have to think through
            break
        rho_history.append(rho)
        rho+=rho_step
        B_history.append([B1,B2])
    
    # Convert cube clusters back to point clusters for output
    B_final = []
    for cube_cluster in remaining_clusters:
        point_cluster = set()
        for point_idx in range(len(data)):
            point_cube = tuple(int((data[point_idx][d]+1) / (2 * delta)) for d in range(dimension))
            if point_cube in cube_cluster:
                point_cluster.add(point_idx)
        if point_cluster:
            B_final.append(point_cluster)
    
    return B_final, rho_history, B_history

#TODO: determine quality of clusters
#TODO: clustering with scipy.cluster
#TODO: determine optimal parameters delta, epsilon, tau

def determine_optimal_delta(data,epsilon_factor,tau_factor):
    Delta=[0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02]
    rho_stars=[]
    for delta in Delta:
        B_final,rho_hist,_=iteration_over_rho(data,delta,epsilon_factor,tau_factor)
        if len(B_final)>1:
            rho_stars.append(rho_hist[-1])
        else: 
            rho_stars.append(float("inf"))
    optimal_delta=Delta[rho_stars.index(min(rho_stars))]
    return optimal_delta





def save_clusters(data,clusters,dimension,dataset_name):
    output=[]
    for cluster_id,cluster in enumerate(clusters):
        for point in cluster:
            output.append({"cluster":cluster_id,"idx":point,"coordinate":[data[point][d] for d in range(dimension)]})
    df=pd.DataFrame(output)
    dataresults_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cluster-results")
    
    df[[f"coordinate_{i}" for i in range(dimension)]]=pd.DataFrame(df.coordinate.tolist(),index=df.index)
    df.assign(freq=df.groupby('cluster')['cluster'].transform('count'))\
  .sort_values(by=['freq','cluster'],ascending=[False,True])
    df[["cluster"]+[f"coordinate_{i}" for i in range(dimension)]].to_csv(os.path.join(dataresults_path,f"team-7-{dataset_name}.result.csv"),index=False,header=False)
        
def save_log(rho_history,B_history,runtime,dataset_name):
    output=[]
    for i,rho in enumerate(rho_history):
        d={"rho":rho}
        d["B0"]=B_history[i][0]
        d["B1"]=B_history[i][1]
        output.append(d)
    df=pd.DataFrame(output)  
    dataresults_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cluster-results")
    
    df.to_csv(os.path.join(dataresults_path,f"team-7-{dataset_name}.log"),index=False,header=False)
    output2=[]
    B1=B_history[-1][0]
    B2=B_history[-1][1]
    output2=[{"Laufzeit":runtime,"B1":B1,"B2":B2,"rho_star":rho_history[-1]}]
    df2=pd.DataFrame(output2)  
    df2.to_csv(os.path.join(dataresults_path,f"team-7-{dataset_name}.result.log"),index=False,header=["Laufzeit","B1","B2","rho_star"])

def plot_clusters(data,clusters,dimension,dataset_name):
    ylorbr = cm.get_cmap('viridis', len(clusters))
    dataresults_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cluster-results")
    
    if dimension==2:
        fig,ax1=plt.subplots(1,1)
        fig2,ax2=plt.subplots(1,1)
        datax,datay=zip(*data)
        ax2.plot(datax,datay,"ro",markersize=2)
        for cluster in clusters:
            cluster_idxs=list(cluster)
            clusterx,clustery=zip(*[(data[idx][0],data[idx][1]) for idx in cluster_idxs])
            ax1.plot(clusterx,clustery,"o",markersize=2,color=ylorbr(clusters.index(cluster)))
            ax1.set_xlim(0,1)
            ax1.set_ylim(0,1)
        fig.savefig(os.path.join(dataresults_path,f"team-7-{dataset_name}.result.png"))
        fig2.savefig(os.path.join(dataresults_path,f"team-7-{dataset_name}.train.png"))
        
    if dimension==3:
        fig=plt.figure()
        fig2=plt.figure()
        ax2=fig.add_subplot(111,projection="3d")
        ax4=fig2.add_subplot(111,projection="3d")
        datax,datay,dataz=zip(*data)
        ax2.scatter(datax,datay,dataz,s=1,alpha=0.1)
        for cluster in clusters:
            cluster_idxs=list(cluster)
            clusterx,clustery,clusterz=zip(*[(data[idx][0],data[idx][1],data[idx][2]) for idx in cluster_idxs])
            ax4.scatter(clusterx,clustery,clusterz,s=1,alpha=0.5,color=ylorbr(clusters.index(cluster)))
            ax4.set_xlim(-0.05,1)
            ax4.set_ylim(-0.05,1)
            ax4.set_zlim(-0.05,1)
        
        fig.savefig(os.path.join(dataresults_path,f"team-7-{dataset_name}.result.png"))
        fig2.savefig(os.path.join(dataresults_path,f"team-7-{dataset_name}.train.png"))