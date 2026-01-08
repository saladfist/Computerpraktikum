#! ./venv/bin/python3
#Verfahren: Zu einem gegebenen Parameter m ∈ N und δ := 1/m wird der Eingaberaum [−1, 1]d zun¨achst
# schachbrettartig in md viele ∥·∥∞-Kugeln A1, . . . , Amd mit Radius δ zerlegt. ¨Uberschneidungen sind somit nur
# an den R¨andern erlaubt, idealerweise werden selbst diese ¨Uberschneidungen aber durch geschicktes Weglassen
# von R¨andern vermieden. Zu einem gegebenen Datensatz D wird dann die Dichtesch¨atzung
# hD,δ (x) := 1
# n2dδd
# md
# X
# j=1
# nX
# i=1
# 1Aj (xi)1Aj (x) , x ∈ [−1, 1]d
# berechnet und f¨ur ein beliebiges ρ ≥ 0 die Menge
# Mρ := x : hD,δ (x) ≥ ρ
# betrachtet. F¨ur ein gegebenes τ > 0 werden anschließend die τ -Zusammenhangskomponenten B1, . . . , BM
# von Mρ bestimmt. Hierbei sind die τ -Zusammenhangskomponenten der Menge Mρ die ¨Aquivalenzklassen
# bez¨uglich der ¨Aquivalenzrelation
# x ∼τ y :⇐⇒ ∃k ≥ 1, y0, . . . , yk ∈ Mρ mit
# y0 = x, yk = y, und ∥yj − yj+1∥ ≤ τ f¨ur alle j = 0, . . . , k − 1 .
# Schließlich werden die Zusammenhangskomponenten Bl eliminiert, f¨ur die
# Bl ∩ xi : hD,δ (xi) ≥ ρ + 2ε = ∅
# gilt. Damit sind f¨ur dieses ρ genau M ′ ≤ M Komponenten ¨ubrig geblieben, die mit ˜B1, . . . , ˜BM ′ bezeichnet
# werden und Cluster genannt werden.
# Zu gegebenem δ > 0 und εfactor ≥ 1 und τfactor ≥ 2 sowie hmax := ∥hD,δ ∥∞ und
# ε := εfactor ·
# r hmax
# n2dδd ,
# τ := τfactor · δ ,
# ρstep := (n2dδd)−1
# iteriert das eigentliche Verfahren nun ¨uber ρ = 0, ρstep, 2ρstep, . . . und stoppt, sowie nach der eben beschrie-
# benen Elimination M ′̸ = 1 gilt. Im Fall M ′ = 0 sollte dann der Datensatz (1, x1), . . . , (1, xn) ausgegeben
# werden, w¨ahrend im Fall M > 1 der Datensatz (y1, x1), . . . , (yn, xn) ausgegeben werden sollte. Hierbei ist
# yi = l f¨ur xi ∈ ˜Bl und yi = 0 falls xi in keinem der Cluster liegt.
# Hinweis: F¨ur fixiertes ρ ≥ 0 ist die Menge Mρ offensichtlich die Vereinigung einiger geeigneter Kugeln
# Ai1 , . . . , Aik . Die obige ¨Aquivalenzrelation induziert nun eine ¨Aquivalenzrelation auf der Menge Ai1 , . . . , Aik ,
# die bei der Berechnung der τ -Zusammenhangskomponenten ausgenutzt werden sollte.
#%%
import matplotlib.pyplot as plt
import pandas as pd
import os 
from collections import deque, defaultdict

def get_coordinate(data,idx,dim):
    return data[idx][dim]

def get_cubes_dict(data,cube_length,dimension):
    #returns a dictionary with cube indices as keys and counts of datapoints in each cube as values
    cubes_dict={} #dictionary to hold the cubes belonging to data points and their counts 
    for point in range(len(data)):
        cube_idx=tuple([int((get_coordinate(data,point,d)+1)/(2*cube_length)) for d in range(dimension)])
        if cube_idx not in cubes_dict:
            cubes_dict[cube_idx]=0
        cubes_dict[cube_idx]+=1 # if data in cube +=1
    return cubes_dict

def get_h_D_delta_unnormalized(x, delta, precomputed_cubes):
    #returns unnormalized h_D,delta(x)
    # Get the cube index for x
    cube_idx_of_x = tuple([int(((x_d)+1)/(2*delta)) for x_d in x])
    # Only count data points in the same cube as x
    h=precomputed_cubes.get(cube_idx_of_x, 0)
    
    return h

def get_distance_sq(data,p1,p2,dimension):
    dist_2=0
    for d in range(dimension):
        dist_2+=(get_coordinate(data,p1,d)-get_coordinate(data,p2,d))**2
    return dist_2
# iteration over thresholds and find M intervals
def get_M(rho,h_dict): #calculates sets Mρ := {x : hD,δ (x) ≥ ρ}
    #M: list of cube indices (as tuples)
    M=[]
    for i,h in h_dict.items():
        # print("i,h",i,h)
        if h>=rho:
            M.append(i)
    return M


def tau_connected_clusters(M,tau,delta,dimension):
    """
    M list of cube indices (as tuples)
    tau float
    dimension int
    Returns: list of clusters, where each cluster is a set of cube indices
    """

    tau_sq = tau**2

    # compute cube center coordinates: center = 2*tau*idx + tau - 1
    def cube_center(c_idx):
        center=tuple(2 * tau * i + tau - 1 for i in c_idx)
        # print("c_idx",c_idx,"center",center)
        return center

    cube_centers = {c: cube_center(c) for c in M}
    def get_neighbors(dimension):
        #returns a list of lists with every distance a cube has to its neighbors in d dimensions
        length=int(tau/(2*delta))
        possible_offsets=range(-length,length+1) #possible entries in x,y,z,... that neighbors of the cube can have to c_idx
        def append_offset(ls):
            new_ls=[]
            for l in ls:
                for i in possible_offsets:
                    new_ls.append(l+[i])
            return new_ls
        offsets=[[]]
        for d in range(dimension):
            offsets=append_offset(offsets)
        return offsets

    neighbors=get_neighbors(dimension) #generate neighbors in each dimension [-1,0,1],[[-1,-1],[-1,0],[-1,1],...]
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!neighbors",neighbors)
    # cluster cubes by connectivity of their centers: center-distance <= tau
    visited_cubes = set()
    clusters = []
    for start_cube in M:
        if start_cube in visited_cubes:
            continue

        stack = [start_cube]
        cluster = set([start_cube])
        while stack:
            c = stack.pop()
            if c in visited_cubes:
                continue
            visited_cubes.add(c)
            c_center = cube_centers[c]
            for n_offset in neighbors:
                neighbor_idx= tuple(c[d]+n_offset[d] for d in range(dimension))
                if neighbor_idx in M and neighbor_idx not in visited_cubes:
                    cluster.add(neighbor_idx) #if neighbor is in M it is automatically tau connected
                    stack.append(neighbor_idx)
            for other in M:
                if other in visited_cubes:
                    continue
                o_center = cube_centers[other]
                dist_sq = 0.0
                for d in range(dimension):
                    dist_sq += (c_center[d] - o_center[d]) ** 2
                    if dist_sq > tau_sq:
                        break
                if dist_sq <= tau_sq and other not in cluster:
                    cluster.add(other)
                    stack.append(other)

        clusters.append(cluster)

    return clusters



def drop_clusters(B,h_dict,rho,epsilon,data,delta,dimension): #drop B if it contains no h with h geq ρ + 2ε; B contains cube indices
    remaining_clusters=[]
    for cluster in B:
        # For each cube in cluster, find max h-value among points in that cube
        max_h = 0
        for point_idx, h in h_dict.items():
            if point_idx in cluster:
                max_h = max(max_h, h)
        if max_h >= rho + 2 * epsilon:
            remaining_clusters.append(cluster)
    return remaining_clusters

def iteration_over_rho(data,delta,epsilon_factor,tau_factor):
    
    dimension=len(data[0])
    cubes_dict = get_cubes_dict(data, delta, dimension)

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
    M_init=get_M(rho,h_dict)
    B_current=tau_connected_clusters(M_init,tau,delta,dimension)
    
    while True:
        remaining_clusters=drop_clusters(B_current,h_dict,rho,epsilon,data,delta,dimension)
        
        if len(remaining_clusters)!=1: #or (M_initial==1 and multiple_clusters): # I think if there exist only 1 cluster B and we dont stop the recursion, then we just increase rho until no clusters remain which leads to large gaps in the clusters, might have to think through
            break
        
        rho_history.append(rho)
        B_history.append(B_current)
        rho+=rho_step
        #update M and B for next iteration
        M_next=get_M(rho,h_dict)
        B_current=tau_connected_clusters(M_next,tau,delta,dimension)
    
    # Convert cube clusters back to point clusters for output
    B_final = []
    for cube_cluster in B_current:
        point_cluster = set()
        for point_idx in range(len(data)):
            point_cube = tuple(int((get_coordinate(data, point_idx, d) + 1) / (2 * delta)) for d in range(dimension))
            if point_cube in cube_cluster:
                point_cluster.add(point_idx)
        if point_cluster:
            B_final.append(point_cluster)
    
    return B_final, rho_history, B_history
    

    
def save_clusters(data,clusters,dimension,dataset_name):
    output=[]
    for cluster_id,cluster in enumerate(clusters):
        for point in cluster:
            output.append({"cluster":cluster_id,"idx":point,"coordinate":[get_coordinate(data,point,d) for d in range(dimension)]})
    df=pd.DataFrame(output)
    df[[f"coordinate_{i}" for i in range(dimension)]]=pd.DataFrame(df.coordinate.tolist(),index=df.index)
    df.assign(freq=df.groupby('cluster')['cluster'].transform('count'))\
  .sort_values(by=['freq','cluster'],ascending=[False,True])
    df[["cluster"]+[f"coordinate_{i}" for i in range(dimension)]].to_csv(f"cluster-results/team-7-{dataset_name}.result.csv",index=False,header=False)
        
def save_log(rho_history,B_history,dataset_name):
    output=[]
    for i,rho in enumerate(rho_history):
        d={"rho":rho}
        for j,cluster in enumerate(B_history[i]):
            d[f"B{j}"]=int(len(cluster))
        output.append(d)
    df=pd.DataFrame(output)  
    df.to_csv(f"cluster-results/team-7-{dataset_name}.log",index=False,header=False)

def plot_clusters(data,clusters,dimension,dataset_name):
    if dimension==2:
        fig,ax1=plt.subplots(1,1)
        fig2,ax2=plt.subplots(1,1)
        datax,datay=zip(*data)
        ax2.plot(datax,datay,"ro",markersize=2)
        for cluster in clusters:
            # if len(cluster)<20:
            #     continue
            cluster_idxs=list(cluster)
            clusterx,clustery=zip(*[(data[idx][0],data[idx][1]) for idx in cluster_idxs])
            # print(clusterdata)
            ax1.plot(clusterx,clustery,"o",markersize=2)
            ax1.set_xlim(0,1)
            ax1.set_ylim(0,1)

        # axs[0].plot(datax,datay,"o",markersize=2)
        fig.savefig(    f"cluster-results/team-7-{dataset_name}.result.png")
        fig2.savefig(   f"cluster-results/team-7-{dataset_name}.train.png")
        # import numpy as np
        # tau_arr=np.arange(0,1,(2**.5*2*0.05))
        # delta_arr=np.arange(0,1,(0.05))
        # X,Y=np.meshgrid(tau_arr,tau_arr)
        # X2,Y2=np.meshgrid(delta_arr,delta_arr)
        # axs[0].plot(X,Y,"ko")
        # axs[0].plot(X2,Y2,"ro")
        # ax.plot(cluster2)
    if dimension==3:
        fig=plt.figure()
        fig2=plt.figure()
        # fig.clf()
        # ax1=fig.add_subplot(111)
        ax2=fig.add_subplot(111,projection="3d")
        # ax3=fig2.add_subplot(223)
        ax4=fig2.add_subplot(111,projection="3d")
        datax,datay,dataz=zip(*data)
        # print("datax",datax)
        # ax1.tricontourf(datax,datay,dataz)
        ax2.scatter(datax,datay,dataz,s=1,alpha=0.1)
        # ax4.scatter(datax,datay,dataz,s=1,alpha=0.05,color="r")
        for cluster in clusters:
            # if len(cluster)<20:
            #     continue
            cluster_idxs=list(cluster)
            clusterx,clustery,clusterz=zip(*[(data[idx][0],data[idx][1],data[idx][2]) for idx in cluster_idxs])
            # print(clusterdata)
            # ax3.tricontourf(clusterx,clustery,clusterz)
            ax4.scatter(clusterx,clustery,clusterz,s=1,alpha=0.5)
            ax4.set_xlim(-0.05,1)
            ax4.set_ylim(-0.05,1)
            ax4.set_zlim(-0.05,1)
        
        fig2.savefig(f"cluster-results/team-7-{dataset_name}.results.png")
        fig.savefig(f"cluster-results/team-7-{dataset_name}.train.png")
# %%
if __name__=="__main__":
    dataset_name="bananas-1-2d"
    df=pd.read_csv(os.path.dirname((os.getcwd()))+f"/data/{dataset_name}.csv",header=None,nrows=10000)
    data=df.values.tolist()
    dimension=len(data[0])

    
    remaining_clusters=iteration_over_rho(data,delta=0.05,epsilon_factor=3,tau_factor=2.00001)
    print(remaining_clusters)
    # save_clusters(remaining_clusters)
    if dimension==2 or dimension==3:
        plot_clusters(data,remaining_clusters,dimension)