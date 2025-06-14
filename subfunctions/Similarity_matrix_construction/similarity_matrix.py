import sys, os
import numpy as np
from mpi4py import MPI      #Importing MPI
#Import packages for geodesic distences
from pyproj import Geod

# get current directory
path = os.getcwd()
# get parent directory
parent_directory = os.path.sep.join(path.split(os.path.sep)[:-2])

# add utils and subfunctions folders to current working path
sys.path.append(parent_directory+"/subfunctions/latlon_transform")
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_construction")

# Import function for the polar rotation
from polar_rotation import polar_rotation_rx 
# Import function to compute pairwise distances between trajectories
from trajectory_distance import integral_vec

def similarity_matrix(Fmap,i_batch,j_batch,K,timemap,geodesic):
    w =  []
    dT = timemap[-1]-timemap[0]
    # Compute time differences
    time_deltas = timemap[1:] - timemap[:-1]
    for k in range(len(i_batch)):
        if (k%int(len(i_batch)/2) == 0):
            print(k)
        if i_batch[k] == j_batch[k] :  #diagonal elements (same trajectory)
            w = np.append(w,K)
        else:
            s=integral_vec(Fmap,dT, time_deltas,i_batch[k],j_batch[k],geodesic)
            if s==0:                     #null disatnce between trajectories
                w = np.append(w,K)
            else:
                w = np.append(w,(1/s))

    #print("number of s = 0 is "+ str(m))
    #print("number of diagonals is "+ str(n))
        
    return w


def similarity_matrix_mpi(Fmap,i_batch,j_batch,K,timemap,geodesic):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0            #The process with rank equal to root will be the one to recieve the results
    #msg = "Hello World! I am process {0} of {1}.\n"
    #sys.stdout.write(msg.format(rank, size))
    #w =  []
    dT = timemap[-1]-timemap[0]
    # Compute time differences
    time_deltas = timemap[1:] - timemap[:-1]

    num_tasks = len(i_batch)                #Total amount of iterations
    start = (rank*num_tasks) // size        #Start index of each process, the // is integer division and will round to the correct number
    end = ((rank+1)*num_tasks) // size      #end index, non-inclusive
    local_w = np.zeros((end-start))         #The w list is replaced with a local array

    """
    For simplicity I have moved trajectory_distance inside similarity_matrix, and changed the order slightly
    The actual calclulations should be the same
    """
    if geodesic==False:
        # Compute pairwise distances at each time step
        for k in range(start,end):
            if i_batch[k] == j_batch[k] :  #diagonal elements (same trajectory)
                local_w[k-start] = K
            else:
                i, j = i_batch[k], j_batch[k]
                distances = np.linalg.norm(Fmap[1:, :, i] - Fmap[1:, :, j], axis=1) + np.linalg.norm(Fmap[:-1, :, i] - Fmap[:-1, :, j], axis=1)
                s = np.sum(time_deltas * distances.reshape(-1))/(dT*2) 
                #s=integral_vec(Fmap,dT, time_deltas,i_batch[k],j_batch[k],geodesic)
                if s==0:                     #null disatnce between trajectories
                    local_w[k-start] = K
                else:
                    local_w[k-start] = 1/s
    else:
        #Loading the Geod and polar rotation are the same for each iteration, moving them outside the loop provides a massive speedup
        # Define the WGS84 ellipsoid
        geod = Geod(ellps='WGS84')  #equivalent to +b=6356752 +a=6378137'
        Fmap_lat,Fmap_lon = polar_rotation_rx(np.array(Fmap[:,1,:]),np.array(Fmap[:,0,:]),-90)  #Fmap[:,0,:] contains longitudes and Fmap[:,1,:] latitudes

        # Compute pairwise distances at each time step
        for k in range(start,end):
            if i_batch[k] == j_batch[k] :  #diagonal elements (same trajectory)
                local_w[k-start] = K
            else:
                i, j = i_batch[k], j_batch[k]
                #Go back to the non-rotated coordinates to compute the geodesic distances
                distances =  geod.inv(Fmap_lon[1:, i], Fmap_lat[1:, i], Fmap_lon[1:, j], Fmap_lat[1:, j])[2] + geod.inv(Fmap_lon[:-1, i], Fmap_lat[:-1, i], Fmap_lon[:-1, j], Fmap_lat[:-1, j])[2]
                s = np.sum(time_deltas * distances.reshape(-1))/(dT*2*1000) # Km
                #s=integral_vec(Fmap,dT, time_deltas,i_batch[k],j_batch[k],geodesic)
                if s==0:                     #null disatnce between trajectories
                    local_w[k-start] = K
                else:
                    local_w[k-start] = 1/s    

    #initializing array w for storing results, only the root will need this
    if rank == root:
        w = np.zeros(num_tasks)
    else:
        w = None

    #print(rank, start, end, i_batch[start], j_batch[start], np.sum(local_w))
    
    
    #Because the different processes may have slightly varying array sizes, the expected sizes of recieved data needs to be specified
    sendcounts = [((((i+1)*num_tasks)//size)-((i*num_tasks)//size)) for i in range(size)]

    """
    The Gatherv method will take all the local arrays and place them in order in the
    w array in the root.
    """
    comm.Gatherv(sendbuf=local_w, recvbuf=(w, sendcounts), root=root)

    #Only the root needs to return anything, as the other processes aren't needed for anything else
    if rank == 0:
        #print(w.shape)
        return w
