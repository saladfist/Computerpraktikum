#! ./venv/bin/python3
#%%
import matplotlib.pyplot as plt
import pandas as pd
import os 
from collections import deque, defaultdict

df=pd.read_csv(os.path.dirname((os.getcwd()))+"/data/toy-2d.csv",header=None)
data=df.values.tolist()
delta_c=0.01
m_cubes=int(1/delta_c)
dimension=len(data[0])

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

cubes_dict=precompute_cubes(data,delta_c,len(data[0]))
h=[]
for x in x_test:
    h.append(h_D_delta(x,data,delta_c,cubes_dict))
plt.tricontourf([x[0] for x in x_test],[x[1] for x in x_test],h,cmap="viridis")
plt.colorbar()
plt.show()

def get_coordinate(idx,dim):
    return data[idx][dim]
def get_distance_sq(p1,p2,dim):
    dist_2=0
    for d in range(dimension):
        dist_2+=(get_coordinate(p1,d)-get_coordinate(p2,d))**2
    return dist_2
# iteration over thresholds and find M intervals
def get_M(data,rho,h_values): #calculates sets Mρ := {x : hD,δ (x) ≥ ρ}
    M=set()
    for i,h in enumerate(h_values):
        if h>=rho:
            M.add(i)
    return list(M)

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
                        dist_2+=(get_coordinate(point,d)-get_coordinate(other_point,d))**2
                    dist=dist_2**.5
                    if dist<=tau:
                        stack.append(other_point)
        B.append(current_cluster)
    return B
""" 
I think a possible better and maybe faster alternative is to decompose the space into (hyper-)cubes
of length tau put every datapoint into a dictionary which maps the index (position) to its cube(-index). 
We then take a random cube and can enter every contained datapoint to a B cluster and query each of the 3**d-1 
neighbors of the cube into a stack and check the distance between points in there to the first cube. Iteratively when 
this is finished and there were points found in the neighboring cubes search these cubes neighbors. cubes 
already searched during this iteration should be exempt due to redundancy (maybe with list of already searched cubes).
When this method runs out of unsearched cubes, choose a next random point not contained in this B.  

"""
def optimized_tau_connected_clusters(M,tau):
    """
    M list
    tau float
    """
    tau_sq=tau**2


    cube_idx_to_point=defaultdict(list)
    point_to_cube_index = {}
    
    def get_cube_idx(point):
        #finds the cube a particular point belongs to 
        return tuple([int(get_coordinate(point,d)/tau) for d in range(dimension)])
    
    for point in M:
        c_idx=get_cube_idx(point)
        cube_idx_to_point[c_idx].append(point) #defaultdict means that if key (c_idx) doesnt exist we create it with corresponding empty list to which we append our values
        point_to_cube_index[point]=c_idx

    offsets=[-1,0,1] #possible entries in x,y,z,... that neighbors of the cube can have to c_idx

    offsets_2d=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]] #all 2d neighbors !!! has to be implemented in d dimensions
    def get_cube_neighbors():
        return
    visited=set()
    B=[]
    for start_point in M:
        if start_point in visited:
            continue #ensures we only start newq clusters from unvisited points

        current_cluster=set()
        point_stack=[start_point] #again deque
        visited_cubes=set()

        while point_stack:
            # print("!!",point_stack)
            curr_point=point_stack.pop()
            # print("!",point_stack,curr_point,visited)
            if curr_point in visited:
                continue
            visited.add(curr_point)
            current_cluster.add(curr_point)
            curr_cube_idx=point_to_cube_index[curr_point]
            
            #get cube neighbors
            cubes_to_check=[curr_cube_idx]+[tuple(curr_cube_idx[d]+offset[d] for d in range(dimension)) for offset in offsets_2d] #will currently not work for d>2

            for cube_idx in cubes_to_check:
                if cube_idx in visited_cubes: #skip if the cube was already prcessed
                    continue
                if cube_idx not in cube_idx_to_point: #skip if cube empty of points
                    continue
                
                for potential_point in cube_idx_to_point[cube_idx]:
                    if potential_point in visited:
                        continue
                    for cluster_point in current_cluster:
                        if get_distance_sq(cluster_point,potential_point,dimension)<=tau_sq:
                            point_stack.append(potential_point)
                            break
                visited_cubes.add(cube_idx)


            visited_cubes.add(curr_cube_idx)
        B.append(current_cluster)
    return B

        


def drop_clusters(B,h_values,data,rho,epsilon): #drop B if it contains no h with h geq ρ + 2ε
    remaining_clusters=[]
    for cluster in B:
        max_h=max([h_values[i] for i in cluster])
        # for point in clusters: #more efficient would be maybe to sort the hvalues of the B set first, if the max h value is smaller we can instantly discard
        #     idx=data.index(list(point)) #seems to be inefficient since it searches the data array linearly everytime maybe lookuptable 
        #     if h_values[idx]>=rho+2*epsilon:
        #         is_higher=True
        #         break
        if max_h>=2*rho+epsilon:
            remaining_clusters.append(cluster)
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
    B_current=optimized_tau_connected_clusters(M_current,tau)
    while True:
        M_initial=len(B_current)
        remaining_clusters=drop_clusters(B_current,h_values,data,rho,epsilon)
        M_new=len(remaining_clusters)
        if M_new==0:
            break
        print(f"rho: {rho}, clusters before drop: {M_initial}, clusters after drop: {M_new})")
        rho+=rho_step
        if rho>1000:
            Exception("Error: rho zu groß")
        #update M and B for next iteration
        M_current=get_M(data,rho,h_values)
        B_current=optimized_tau_connected_clusters(M_current,tau)
    
    return B_current



remaining_clusters=0
remaining_clusters=iteration_over_rho(data,delta=0.05,epsilon_factor=3,tau_factor=5.000001)
print(remaining_clusters)
#plot clusters
#%%
fig,axs=plt.subplots(1,2)
print(data)
datax,datay=zip(*data)
print(len(remaining_clusters))
for cluster in remaining_clusters:
    print(list(cluster))
    cluster_idxs=list(cluster)
    clusterx,clustery=zip(*[(data[idx][0],data[idx][1]) for idx in cluster_idxs])
    # print(clusterdata)
    axs[0].plot(clusterx,clustery,"o",markersize=2)
    axs[0].set_xlim(0,1)
    axs[0].set_ylim(0,1)


axs[1].plot(datax,datay,"o",markersize=2)

# ax.plot(cluster2)
