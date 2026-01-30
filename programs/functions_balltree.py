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
    
    def cube_center(c_idx):
        return tuple(-1 + (i + 0.5) * cube_length for i in c_idx)
    
    cubes_centers = {c: cube_center(c) for c in cubes_dict.keys()}
    return cubes_dict,cubes_point_map, cubes_centers

def get_M(rho,h_dict): #calculates sets Mρ := {x : hD,δ (x) ≥ ρ}
    #M: list of cube indices (as tuples)
    #not_M: list of cube indices (as tuples)
    M=[]
    not_M=[]
    for i,h in h_dict.items():
        # print("i,h",i,h)
        if h>=rho:
            M.append(i)
        else:
            not_M.append(i)
            
    return M,not_M

def tau_connected_clusters(M,tau,delta,dimension,cubes_dict): #-data - cube_centers
    """
    M list of cube indices (as tuples)
    tau float
    delta float
    dimension int
    Returns: list of clusters, where each cluster is a set of cube indices
    """

    # cluster cubes by connectivity of their centers: 
    # center-distance <= tau -> connected 
    # center-distance <= tau+2delta -> check whether these cubes contain points close enough to be connected 
    threshold = (tau)/(2*delta)+1  
    def cubes_connected(c1,c2):
        dist=[c1[d]-c2[d] for d in range(dimension)]
        dist_abs=max(abs(_) for _ in dist)
        if dist_abs<threshold:
            return True
        return False
    
    visited = set()
    clusters = []
    M_set = set(M)
    for start_cube in M_set:
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
            
            for other in M_set:
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

def backwards_rho_iteration(data, delta, epsilon_factor, tau_factor):
    dimension = len(data[0])
    cubes_dict, cubes_point_map, cubes_centers = get_cubes_dict(data, 2*delta, dimension)
    
    # Calculate h_values for all data points
    n = len(data)
    data_dict = {}
    h_dict = {}
    total_volume = n * (2*delta)**dimension
    for i, data_point in enumerate(data):
        data_dict[i] = {"cluster": 0, "idx": i, "coordinate": data_point}
    for cube_idx, count in cubes_dict.items():
        h_dict[cube_idx] = count / total_volume
    h_max = max(h_dict.values())
    
    epsilon = epsilon_factor * (h_max / (n * (2*delta)**dimension))**0.5
    tau = tau_factor * delta
    rho_step = 1 / (n * (2*delta)**dimension)
    
    cubes_by_h = sorted(h_dict.items(), key=lambda x: -x[1])
    cubes_by_rho = defaultdict(list)
    for cube, rho in cubes_by_h:
        cubes_by_rho[rho].append(cube)
    cubes_by_rho = sorted(cubes_by_rho.items(), key=lambda x: -x[0])
    
    parent = {}
    clusters = {}
    active_cubes = set()
    cluster_max_h = {}
    cluster_length = {}
    last_valid_clusters = None
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            clusters[ra] |= clusters[rb]
            cluster_max_h[ra] = max(cluster_max_h[ra], cluster_max_h[rb])
            cluster_length[ra] += cluster_length[rb]
            del clusters[rb]
            del cluster_max_h[rb]
            del cluster_length[rb]
    
    rho_history = []
    B_history = []
    threshold = (tau / (2*delta)) + 1
    
    def cubes_connected(c1, c2):
        for d in range(dimension):
            diff = abs(c1[d] - c2[d])
            if diff >= threshold:
                return False
        return True
    
    # True KD-Tree implementation
    class ChebyshevKDTree:
        def __init__(self, dimension):
            self.dim = dimension
            self.root = None
            self.cube_to_node = {}  # Map cube to its node for quick access
        
        class Node:
            def __init__(self, cube, depth):
                self.cube = cube
                self.left = None
                self.right = None
                self.depth = depth
                self.axis = depth % dimension
        
        def insert(self, cube):
            self.root = self._insert(self.root, cube, 0)
            self.cube_to_node[cube] = self.root
        
        def _insert(self, node, cube, depth):
            if node is None:
                return self.Node(cube, depth)
            
            axis = depth % self.dim
            if cube[axis] < node.cube[axis]:
                node.left = self._insert(node.left, cube, depth + 1)
            else:
                node.right = self._insert(node.right, cube, depth + 1)
            
            return node
        
        def find_candidates(self, query_cube, max_diff):
            """Find candidate cubes that could be within Chebyshev distance."""
            candidates = []
            if self.root:
                self._find_candidates_recursive(self.root, query_cube, max_diff, candidates, 0)
            # Remove query cube if present and remove duplicates
            return list(set(c for c in candidates if c != query_cube))
        
        def _find_candidates_recursive(self, node, query_cube, max_diff, candidates, best_dim):
            """Recursively find candidate cubes using KD-tree pruning."""
            if node is None:
                return
            
            # Check if this node could be a candidate based on best dimension heuristic
            # Similar to original algorithm: use dimension with smallest range first
            if best_dim >= self.dim:
                best_dim = 0
            
            axis_diff = abs(query_cube[best_dim] - node.cube[best_dim])
            
            # If difference in best dimension is within range, add as candidate
            if axis_diff < max_diff:
                candidates.append(node.cube)
            
            # Decide which branches to search based on splitting plane
            axis = node.axis
            diff = query_cube[axis] - node.cube[axis]
            
            # Search the side that contains the query point first
            if diff <= 0:
                self._find_candidates_recursive(node.left, query_cube, max_diff, candidates, best_dim + 1)
                # If query point is close to splitting plane, search other side too
                if abs(diff) < max_diff:
                    self._find_candidates_recursive(node.right, query_cube, max_diff, candidates, best_dim + 1)
            else:
                self._find_candidates_recursive(node.right, query_cube, max_diff, candidates, best_dim + 1)
                if abs(diff) < max_diff:
                    self._find_candidates_recursive(node.left, query_cube, max_diff, candidates, best_dim + 1)
    
    # Initialize KD-Tree
    kd_tree = ChebyshevKDTree(dimension)
    
    # Process cubes in decreasing density order
    for rho, cubes in cubes_by_rho:
        for cube in cubes:
            # Activate cube
            active_cubes.add(cube)
            parent[cube] = cube
            clusters[cube] = {cube}
            cluster_max_h[cube] = h_dict[cube]
            cluster_length[cube] = cubes_dict[cube]
            
            # Insert cube into KD-Tree
            kd_tree.insert(cube)
            
            # Find candidate neighbors using KD-Tree
            candidates = kd_tree.find_candidates(cube, threshold)
            
            # Check connection and union
            for other in candidates:
                if other in active_cubes and cubes_connected(cube, other):
                    union(cube, other)
        
        # Current clusters
        remaining_clusters = [
            cl for cl in clusters.values()
            if cluster_max_h[find(next(iter(cl)))] >= rho + epsilon
        ]
        
        rho_history.append(rho)
        lengths = sorted((cluster_length[find(next(iter(cl)))] for cl in remaining_clusters), reverse=True)
        B1 = lengths[0] if len(lengths) > 0 else 0
        B2 = lengths[1] if len(lengths) > 1 else 0
        B_history.append([B1, B2])
        
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
                for point_idx in cubes_point_map.get(cube, []):
                    data_dict[point_idx]["cluster"] = cid
    
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
