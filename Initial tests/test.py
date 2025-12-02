#! ./venv/bin/python3
#%%
import matplotlib.pyplot as plt
import pandas as pd
import os 
import numpy as np

data=pd.read_csv(os.path.dirname((os.getcwd()))+"/data/bananas-1-2d.csv",header=None)
data_list=data.values.tolist()
delta_c=0.01
m_cubes=int(1/delta_c)
dimension=len(data_list[0])

def get_cube_index(point,delta,d=dimension):
    m=1/delta
    indices=[]
    #iterate over dimensions to find nearest cube
    for dim in range(d):
        coord_d=point[dim] # get coordinate in dim d 
        index=int(coord_d/delta)
        index=min(index,m-1) # handle boundary
        indices.append(index)
    return tuple(indices)

def precompute_cubes(data,delta,d=dimension):
    m=1/delta
    cubes_dict={} #dictionary to hold the cubes belonging to data points and their counts 
    for i,point in enumerate(data):
        cube_idx=get_cube_index(point,delta,d)
        if cube_idx not in cubes_dict:
            cubes_dict[cube_idx]=0
        cubes_dict[cube_idx]+=1 # if data in cube +=1
    return cubes_dict

def h_D_delta(x, data, delta, precomputed_cubes=None):
    m=1/delta
    n=len(data)
    d=len(data[0])
    
    if precomputed_cubes is None:
        precomputed_cubes=precompute_cubes(data, delta,d)
    # Get the cube index for x
    x_cube =get_cube_index(x, delta, d)
    # Only count data points in the same cube as x
    h=precomputed_cubes.get(x_cube, 0)
    
    h/=n*2**d*delta**d
    return h

x_test=[]
for i in range(m_cubes+1):
    for j in range(m_cubes+1):
        x_test.append([i/m_cubes,j/m_cubes])

cubes_dict=precompute_cubes(data_list,delta_c,len(data_list[0]))
h=[]
for x in x_test:
    h.append(h_D_delta(x,data_list,delta_c,cubes_dict))
plt.tricontourf([x[0] for x in x_test],[x[1] for x in x_test],h,cmap="viridis")
plt.colorbar()
plt.show()

# iteration over thresholds and find M intervals
def get_M(data,rho,h_values): #calculates sets Mρ := {x : hD,δ (x) ≥ ρ}
    M=set()
    for i,h in enumerate(h_values):
        if h>=rho:
            M.add(tuple(data[i]))
    return M

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
            point=stack.pop() #removes point from stack, if now point found breaks loop
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
                        dist_2+=(point[d]-other_point[d])**2
                    dist=dist_2**.5
                    if dist<=tau:
                        stack.append(other_point)
        B.append(current_cluster)
    return B


def drop_clusters(B,h_values,data,rho,epsilon): #drop B if it contains no h with h geq ρ + 2ε
    remaining_clusters=[]
    for clusters in B:
        max_h=max([h_values[data.index(list(point))] for point in clusters])
        # for point in clusters: #more efficient would be maybe to sort the hvalues of the B set first, if the max h value is smaller we can instantly discard
        #     idx=data.index(list(point)) #seems to be inefficient since it searches the data array linearly everytime maybe lookuptable 
        #     if h_values[idx]>=rho+2*epsilon:
        #         is_higher=True
        #         break
        if max_h>=2*rho+epsilon:
            remaining_clusters.append(clusters)
    return remaining_clusters

def iteration_over_rho(data,delta,epsilon_factor,tau_factor):
    n=len(data)
    d=len(data[0])
    cubes_dict=precompute_cubes(data,delta,d)
    h_values=[]
    rho=0

    for x in data:
        h_values.append(h_D_delta(x,data,delta,cubes_dict))
        h_max=max(h_values)
        epsilon=epsilon_factor*(h_max/(n*2**d*delta**d))**.5
        tau=tau_factor*delta
        rho_step=(n*2**d*delta**d)**-1
    # find equivalence classes and clusters
    M_current=get_M(data,rho,h_values)
    B_current=get_tau_connected_clusters(M_current,tau)
    while True:
        M_initial=len(B_current)
        remaining_clusters=drop_clusters(B_current,h_values,data,rho,epsilon)
        M_new=len(remaining_clusters)
        if M_new==0:
            break
        print(f"rho: {rho}, clusters before drop: {M_initial}, clusters after drop: {M_new})")
        rho+=rho_step
        if rho>1000:
            Exception("error")
        #update M and B for next iteration
        M_current=get_M(data,rho,h_values)
        B_current=get_tau_connected_clusters(M_current,tau)
    return B_current




remaining_clusters=iteration_over_rho(data_list,delta=0.05,epsilon_factor=3.0,tau_factor=2.000001)
print(remaining_clusters)
#plot clusters
#%%
fig,axs=plt.subplots(1,2)
print(data_list)
datax,datay=zip(*data_list)
print(len(remaining_clusters))
for cluster in remaining_clusters:
    print(cluster)
    clusterx,clustery=zip(*list(cluster))
    axs[0].plot(clusterx,clustery,"o",markersize=2)


axs[1].plot(datax,datay,"o",markersize=2)

# ax.plot(cluster2)
