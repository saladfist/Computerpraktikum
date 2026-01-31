import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import os 
from collections import deque, defaultdict
from bisect import bisect_left, bisect_right, insort


def get_cubes_dict(data,cube_length,dimension):
    '''
    data: list of data points
    cube_length: float 2delta
    dimension: float dimension of data
    returns:
    #cubes_dict: a dictionary with cube indices as keys and counts of datapoints in each cube as values
    #cubes_point_map: a dictionary with each cube containing points with all data point indices in that cube
    #cubes_coords: a dictionary with each cube middle point coordinates as values
    '''
    cubes_dict={} #dictionary to hold the cubes belonging to data points and their counts 
    cubes_point_map=defaultdict(list)
    cells_per_dim = int(2 / cube_length)

    for point in range(len(data)):
        cube_idx_list = []
        for d in range(dimension):
            idx = int((data[point][d] + 1) / cube_length)
            
            if idx >= cells_per_dim:  #handle values that are very close to the edge
                idx = cells_per_dim - 1
            if idx < 0:
                idx = 0
            cube_idx_list.append(idx)
        cube_idx = tuple(cube_idx_list)
        
        if cube_idx not in cubes_dict:
            cubes_dict[cube_idx] = 0 
        cubes_dict[cube_idx] += 1
        cubes_point_map[cube_idx].append(point)
        
    return cubes_dict,cubes_point_map
def backwards_rho_iteration(data,delta,epsilon_factor,tau_factor):
    dimension=len(data[0])
    cubes_dict,cubes_point_map = get_cubes_dict(data, 2*delta, dimension)

    #calulate h_values for all data points
    n=len(data)
    data_dict={}
    h_dict = {}
    total_volume = n * (2*delta)**dimension
    h_dict = {cube: count / total_volume for cube, count in cubes_dict.items()}
    h_max = max(h_dict.values())
    epsilon=epsilon_factor*(h_max/(n*(2*delta)**dimension))**.5
    tau=tau_factor*delta

    cubes_by_rho = defaultdict(list)
    for cube, rho in h_dict.items():
        cubes_by_rho[rho].append(cube)
    cubes_by_rho = sorted(cubes_by_rho.items(), key=lambda x: -x[0])

    parent={}
    clusters={}
    active_cubes=set()
    cluster_max_h = {} #caching max_h
    cluster_length = {} #caching B1 and B2

    def find(x): #get representative of cluster
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            # Merge cluster sets
            clusters[ra] |= clusters[rb]
            cluster_max_h[ra] = max(cluster_max_h[ra], cluster_max_h[rb])
            cluster_length[ra] += cluster_length[rb]
            # Clean up merged cluster
            del clusters[rb]
            del cluster_max_h[rb]
            del cluster_length[rb]

    def cubes_connected(c1, c2):
        for d in range(dimension):
            diff = abs(c1[d] - c2[d])
            if diff >= threshold:
                return False
        return True

    def find_possible_neighbors(cube, sorted_lists):
        if not sorted_lists:
            return []
        best_candidates = []
        min_count = float('inf')
        min_idx=0
        max_idx=len(sorted_lists[0])
        #find best dimension
        for d in range(dimension):
            lst = sorted_lists[d]
            low = max(cube[d] - threshold,min_idx)
            high = min(cube[d] + threshold,max_idx)
            start = bisect_left(lst, low, key=lambda tup: tup[d])
            end = bisect_right(lst, high, key=lambda tup: tup[d])
            count = end - start
            if count < min_count:
                min_count = count
                best_candidates = lst[start:end]

        if not best_candidates:
            return []
        neighbors = []
        for other in best_candidates:
            if cubes_connected(cube,other):
                neighbors.append(other)
        return neighbors

    # Possibly we can do this recursively, by starting from rho_max/iteration_depth and checking whether we get len(remaining_clusters)!=1 and if not starting from rho_max(i+1)/iteration_depth etc...
    sorted_active_cubes = [[] for _ in range(dimension)]
    last_valid_clusters = None
    rho_history=[]
    B_history=[]
    threshold = (tau / (2*delta) + 1)

    for rho, cubes in cubes_by_rho:
        for cube in cubes:
            # activate cubes
            active_cubes.add(cube)
            parent[cube] = cube#
            clusters[cube] = {cube}
            cluster_max_h[cube] = h_dict[cube]
            cluster_length[cube] = cubes_dict[cube]

            # insert cube into sorted lists
            for d in range(dimension):
                insort(sorted_active_cubes[d], cube, key=lambda tup: tup[d])

            # union with neighbors
            for other in find_possible_neighbors(cube, sorted_active_cubes):
                union(cube, other)

        # Current clusters
        remaining_clusters = [
            cluster_set for cluster_set in clusters.values()
            if cluster_max_h[find(next(iter(cluster_set)))] >= rho + epsilon
        ]

        rho_history.append(rho)
        lengths = sorted(
            (cluster_length[find(next(iter(cl)))] for cl in remaining_clusters),
            reverse=True
        )
        B1 = lengths[0] if len(lengths) > 0 else 0
        B2 = lengths[1] if len(lengths) > 1 else 0
        B_history.append([B1,B2])

        # Store last nontrivial clustering
        if len(remaining_clusters) != 1:
            last_valid_clusters = remaining_clusters
            rho_history = [rho]
            B_history = [[B1, B2]]

    # Assign clusters
    data_dict = {
        i: {"cluster": 0, "idx": i, "coordinate": data[i]}
        for i in range(len(data))
    }

    if last_valid_clusters is not None:
        for cid, cube_cluster in enumerate(last_valid_clusters, start=1):
            for cube in cube_cluster:
                for idx in cubes_point_map[cube]:
                    data_dict[idx]["cluster"] = cid

    return data_dict, rho_history[::-1], B_history[::-1]
#TODO: determine quality of clusters
#TODO: determine optimal parameters delta, epsilon, tau

def determine_optimal_delta(data,epsilon_factor,tau_factor):
    Delta=[0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02]
    rho_stars=[]
    for delta in Delta:
        data_dict,rho_hist,_=backwards_rho_iteration(data,delta,epsilon_factor,tau_factor)
        clusters=pd.DataFrame(data_dict.values()).groupby("cluster")["idx"].apply(list).tolist()
        if len(clusters)>1:
            rho_stars.append(rho_hist[-1])
        else: 
            rho_stars.append(float("inf"))
    optimal_delta=Delta[rho_stars.index(min(rho_stars))]
    return optimal_delta


def save_clusters(df,dimension,dataset_name):
    dataresults_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cluster-results")
    
    df[[f"coordinate_{i}" for i in range(dimension)]]=pd.DataFrame(df.coordinate.tolist(),index=df.index)
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
    df2.to_csv(os.path.join(dataresults_path,f"team-7-{dataset_name}.result.log"),index=False,header=False)

def plot_clusters(df,dimension,dataset_name,kmeans_used=False):
    dataresults_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cluster-results")
    add_name="_kmeans" if kmeans_used else ""
    
    clusters=df[df["cluster"]!=0].groupby("cluster")["idx"].apply(list).tolist()
    unclustered_points=df[df["cluster"]==0]["idx"].tolist()
    data=df[["coordinate"]].values.tolist()
    data=[data[i][0] for i in range(len(data))]
    ylorbr = cm.get_cmap('viridis', len(clusters))
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
        ax1.plot([data[idx][0] for idx in unclustered_points],[data[idx][1] for idx in unclustered_points],"ko",markersize=2,alpha=0.1)
        fig.savefig(os.path.join(dataresults_path,f"team-7-{dataset_name}.result"+add_name+".png"),dpi=500)
        fig2.savefig(os.path.join(dataresults_path,f"team-7-{dataset_name}.train"+add_name+".png"),dpi=500)
        
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
        ax4.scatter([data[idx][0] for idx in unclustered_points],
                    [data[idx][1] for idx in unclustered_points],
                    [data[idx][2] for idx in unclustered_points],
                    "o",color="k",s=1,alpha=0.1)
            
        
        fig.savefig(os.path.join(dataresults_path,f"team-7-{dataset_name}.train"+add_name+".png"),dpi=500)
        fig2.savefig(os.path.join(dataresults_path,f"team-7-{dataset_name}.result"+add_name+".png"),dpi=500)
