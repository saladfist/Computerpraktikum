#! ./venv/bin/python3
#%%
import matplotlib.pyplot as plt
import pandas as pd
import os 
from collections import deque, defaultdict
from  numba import njit

delta_c=0.01
m_cubes=int(1/delta_c)
dimension=len(data[0])


x_test=[]
for i in range(m_cubes+1):
    for j in range(m_cubes+1):
        x_test.append([i/m_cubes,j/m_cubes])

# cubes_dict=precompute_cubes(data,delta_c,len(data[0]))
# h=[]
# for x in x_test:
#     h.append(h_D_delta(x,data,delta_c,cubes_dict))
# plt.tricontourf([x[0] for x in x_test],[x[1] for x in x_test],h,cmap="viridis")
# plt.colorbar()
# plt.show()


def get_cube_index(point,delta,d=dimension):
    m=int(1/(delta)) #from 0 to 1 so each cube has side length delta
    indices=[]
    #iterate over dimensions to find nearest cube
    for dim in range(d):
        coord_d=point[dim] # get coordinate in dim d 
        index=int(coord_d/delta)
        index=min(max(index,0),m-1) # handle boundary
        indices.append(index)
    return tuple(indices)

def precompute_cubes(data,delta,d=dimension):
    cubes_dict={} #dictionary to hold the cubes belonging to data points and their counts 
    for point in data:
        cube_idx=get_cube_index(point,delta,d)
        if cube_idx not in cubes_dict:
            cubes_dict[cube_idx]=0
        cubes_dict[cube_idx]+=1 # if data in cube +=1
    return cubes_dict

def h_D_delta(x, data, delta, precomputed_cubes=None):
    n=len(data)
    d=len(data[0])    
    if precomputed_cubes is None:
        precomputed_cubes=precompute_cubes(data, delta,d)
    # Get the cube index for x
    x_cube =get_cube_index(x, delta, d)
    # Only count data points in the same cube as x
    h=precomputed_cubes.get(x_cube, 0)
    
    h=h/(n*(delta)**d)
    return h
def get_coordinate(data,idx,dim):
    return data[idx][dim]
def get_distance_sq(data,p1,p2,dim):
    dist_2=0
    for d in range(dim):
        dist_2+=(get_coordinate(data,p1,d)-get_coordinate(data,p2,d))**2
    return dist_2
# iteration over thresholds and find M intervals
def get_M(rho,h_values): #calculates sets Mρ := {x : hD,δ (x) ≥ ρ}
    M=set()
    for i,h in enumerate(h_values):
        if h>=rho:
            M.add(i)
    return list(M)

njit(parallel=True)
def get_tau_connected_clusters(M,tau): #get B sets 
    visited=set()
    B=[]
    M_list = list(M)
    for start_point in M_list:
        if start_point in visited:
            continue #ensures we only start newq clusters from unvisited points
        current_cluster=set()
        #stacks
        stack=[start_point]
        while stack: #while there are points in the stack, we can explore new paths else breaks
            point=stack.pop() #removes point from stack, if now point found breaks loop maybe more efficient with an actual stack which can faster delete ends f.e "collections.deque"
            if point in visited:
                continue
            visited.add(point)
            current_cluster.add(point)
            #find neighbors with distance leq tau
            for other_point in M_list:
                if other_point not in visited:
                    #dist=||point-other_point|| maybe more efficient caluclation possivble
                    dist_2=0
                    for d in range(dimension):
                        dist_2+=(get_coordinate(data,point,d)-get_coordinate(data,other_point,d))**2
                    dist=dist_2**.5
                    if dist<=tau:
                        stack.append(other_point)
        B.append(current_cluster)
    return B

njit(parallel=True)
def optimized_tau_connected_clusters(M,tau):
    """
    M list
    tau float
    """
    tau_sq=tau**2

    cube_idx_to_point=defaultdict(list)
    point_to_cube_index = {}
    
    def get_coordinate(data,idx,dim):
        return data[idx][dim]
    njit(parallel=True)
    def get_cube_idx(point):
        #finds the cube a particular point belongs to 
        return tuple([int(get_coordinate(data,point,d)/(tau)) for d in range(dimension)])
    
    njit(parallel=True)
    def get_cube_neighbors(dimension):
        #returns a list of lists with every distance a cube has to its neighbors in d dimensions
        possible_offsets=[-1,0,1] #possible entries in x,y,z,... that neighbors of the cube can have to c_idx
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
    
    for point in M:
        c_idx=get_cube_idx(point)
        cube_idx_to_point[c_idx].append(point) #defaultdict means that if key (c_idx) doesnt exist we create it with corresponding empty list to which we append our values
        point_to_cube_index[point]=c_idx

    offsets=get_cube_neighbors(dimension)
    visited=set()
    B=[]
    for start_point in M:
        if start_point in visited:
            continue #ensures we only start newq clusters from unvisited points

        current_cluster=set()
        point_stack=[start_point] #again we can use actual stack maybe
        
        while point_stack:
            curr_point=point_stack.pop() #remove current point from stack
            visited.add(curr_point)
            current_cluster.add(curr_point) #add current point to current cluster
            curr_cube=point_to_cube_index[curr_point]
            for offset in offsets:
                nghbr_cube=tuple(curr_cube[d]+offset[d] for d in range(dimension))
                if nghbr_cube not in cube_idx_to_point: #skips if cube contains no points
                    continue
                potential_points=cube_idx_to_point[nghbr_cube]
                for point in potential_points:
                    if point not in visited and get_distance_sq(data,curr_point,point,dimension)<=tau_sq :
                        visited.add(point)
                        point_stack.append(point)
                        
        B.append(current_cluster)
    return B

def drop_clusters(B,h_values,rho,epsilon): #drop B if it contains no h with h geq ρ + 2ε
    remaining_clusters=[]
    for cluster in B:
        max_h=max([h_values[i] for i in cluster])
        if max_h>=rho+2*epsilon:
            remaining_clusters.append(cluster)
    return remaining_clusters
njit(parallel=True)
def iteration_over_rho(data,delta,epsilon_factor,tau_factor):
    n=len(data)
    d=len(data[0])
    h_values=[]
    rho=0
    
    
    h_max=0
    for x in data:
        h_values.append(h_D_delta(x,data,delta))
        h_max=max(h_max,max(h_values))
    epsilon=epsilon_factor*(h_max/(n*delta**d))**.5
    tau=tau_factor*delta
    rho_step=1/(n*(delta)**d)

    # find equivalence classes and clusters
    M_current=get_M(rho,h_values)
    B_current=optimized_tau_connected_clusters(M_current,tau)
    print("B_Current",B_current)
    while True:
        M_initial=len(B_current)
        remaining_clusters=drop_clusters(B_current,h_values,rho,epsilon)
        M_new=len(remaining_clusters)
        
        if M_new!=1: #or (M_initial==1 and multiple_clusters): # I think if there exist only 1 cluster B and we dont stop the recursion, then we just increase rho until no clusters remain which leads to large gaps in the clusters, might have to think through
            break
        print(f"rho: {rho}, clusters before drop: {M_initial}, clusters after drop: {M_new})")
        rho+=rho_step
        if rho>1000:
            Exception("Error: rho zu groß")
        #update M and B for next iteration
        M_current=get_M(rho,h_values)
        B_current=optimized_tau_connected_clusters(M_current,tau)
    
    return B_current
#%%
dataset_name="toy-2d"
df=pd.read_csv(os.path.dirname((os.getcwd()))+f"/data/{dataset_name}.csv",header=None,nrows=2000)
data=df.values.tolist()

remaining_clusters=iteration_over_rho(data,delta=0.05,epsilon_factor=3,tau_factor=0.800001)
print(remaining_clusters)
#plot clusters
def save_clusters(clusters):
    output=[]
    for cluster_id,cluster in enumerate(clusters):
        for point in cluster:
            output.append({"cluster":cluster_id,"idx":point,"coordinate":[get_coordinate(data,point,d) for d in range(dimension)]})
    df=pd.DataFrame(output)
    df[[f"coordinate_{i}" for i in range(dimension)]]=pd.DataFrame(df.coordinate.tolist(),index=df.index)
    df.assign(freq=df.groupby('cluster')['cluster'].transform('count'))\
  .sort_values(by=['freq','cluster'],ascending=[False,True])
    # print(df)
    df.to_csv(f"out/{dataset_name}.csv")
    # for cluster_id in range(len(clusters)):
    df[[f"coordinate_{i}" for i in range(dimension)]+["cluster"]].to_csv(f"out/ggobi_{dataset_name}.csv",index=False,header=False)
    
save_clusters(remaining_clusters)
# print(len(remaining_clusters),len(remaining_clusters[0]),len(remaining_clusters[1]))

#%%
if dimension==2:
    fig,axs=plt.subplots(1,2)
    print(data)
    datax,datay=zip(*data)
    print(len(remaining_clusters))
    # axs[1].plot(datax,datay,"ro",markersize=2)
    for cluster in remaining_clusters:
        # if len(cluster)<20:
        #     continue
        print(list(cluster))
        cluster_idxs=list(cluster)
        clusterx,clustery=zip(*[(data[idx][0],data[idx][1]) for idx in cluster_idxs])
        # print(clusterdata)
        axs[1].plot(clusterx,clustery,"o",markersize=2)
        axs[1].set_xlim(0,1)
        axs[1].set_ylim(0,1)

    axs[0].plot(datax,datay,"o",markersize=2)
    # import numpy as np
    # tau_arr=np.arange(0,1,(2**.5*2*0.05))
    # delta_arr=np.arange(0,1,(0.05))
    # X,Y=np.meshgrid(tau_arr,tau_arr)
    # X2,Y2=np.meshgrid(delta_arr,delta_arr)
    # axs[0].plot(X,Y,"ko")
    # axs[0].plot(X2,Y2,"ro")
    # ax.plot(cluster2)
if dimension==3:
    print(data)
    fig=plt.figure()
    # fig.clf()
    ax1=fig.add_subplot(221)
    ax2=fig.add_subplot(222,projection="3d")
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224,projection="3d")
    datax,datay,dataz=zip(*data)
    print("datax",datax)
    ax1.tricontourf(datax,datay,dataz)
    ax2.scatter(datax,datay,dataz,s=1,alpha=0.1)
    # ax4.scatter(datax,datay,dataz,s=1,alpha=0.05,color="r")
    for cluster in remaining_clusters:
        if len(cluster)<20:
            continue
        print(list(cluster))
        cluster_idxs=list(cluster)
        clusterx,clustery,clusterz=zip(*[(data[idx][0],data[idx][1],data[idx][2]) for idx in cluster_idxs])
        # print(clusterdata)
        ax3.tricontourf(clusterx,clustery,clusterz)
        ax4.scatter(clusterx,clustery,clusterz,s=1,alpha=0.5)
        ax4.set_xlim(-0.05,1)
        ax4.set_ylim(-0.05,1)
        ax4.set_zlim(-0.05,1)
# %%
