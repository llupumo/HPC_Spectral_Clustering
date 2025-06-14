import netCDF4 as nc
import sys, os, argparse
import time
import numpy as np
from numpy import ma as ma

from mpi4py import MPI      #Importing MPI

#Import packages for plotting
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from pylab import imshow,cm

#Import packages for clustering
from sklearn.cluster import KMeans
from scipy.linalg import eigh

#Import packages for geodesic distences
from pyproj import Geod


#Import packages for interpolating and filtering data
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.interpolate import LinearNDInterpolator as LNDI

# Import package for parallel computing
from joblib import Parallel, delayed

# Create the parser
parser = argparse.ArgumentParser(description="Process some parameters for clustering.")
# Add required arguments
parser.add_argument("Ncores", type=int, help="Number of CPU's")
parser.add_argument("input_files_directory", type=str, help="Directory to the input files")
parser.add_argument("parent_directory", type=str, help="Parent directory")
parser.add_argument("results_directory", type=str, help="Results directory")
parser.add_argument("geodesic", type=lambda x: x.lower() == 'true', help="Geodesic boolean for trajectory distance")
parser.add_argument("tmin", type=int, help="Results directory")
# Add optional argument with a default value
parser.add_argument("--K", type=int, default=1000, help="K similarity diagonal (default: 1000)")
# Parse the arguments
args = parser.parse_args()

Ncores = args.Ncores
input_files_directory = args.input_files_directory
parent_directory = args.parent_directory
results_directory = args.results_directory
geodesic = args.geodesic
tmin = args.tmin
K = args.K
time_steps_per_day=4

W_params = (
    f"geodesic_{geodesic}"
)

# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_construction")
sys.path.append(parent_directory+"/subfunctions/latlon_transform")
sys.path.append(parent_directory+"/utils")

#from trajectory_distance import integral_vec
#from similarity_matrix import similarity_matrix        #Commented out since they're currently in this script
from polar_rotation import polar_rotation_rx

def similarity_matrix(Fmap,i_batch,j_batch,K,timemap,geodesic):
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
                s = np.sum((time_deltas / 2) * distances)/dT
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
                s = np.sum((time_deltas / 2) * distances)/dT
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


####################################################################################

#Needed for ensuring only the root starts saving arrays and images
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

#Read input data
Fmap_path = input_files_directory+'/'+str(int(tmin/time_steps_per_day))+'_Fmap_matrix.npy'
time_path = input_files_directory+'/'+str(int(tmin/time_steps_per_day))+'_advection_time.npy'
# Load the Fmap array from the file
Fmap = np.load(Fmap_path)  # ntime [lon,lat] ntrajectories
# Load the time_adv_mod array from the file
time_adv_mod = np.load(time_path)



print("Preparing the parallel loop to compute the Similarity matrix for the day "+str(tmin)) 
n = Fmap.shape[2]
print(f"{n} trajectories are being processed. Each trajectory has {Fmap.shape[0]} time steps.")

indices = np.tril_indices(n,0,n)

I=indices[0]
J=indices[1]

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

#We do the splitting inside similarity_matrix, not here
I_batch = I #list(split(I, Ncores)) # list (Nx*Ny)
J_batch = J #list(split(J, Ncores)) # list (Nx*Ny)


print("Number of elements in W triangular:")
print(n*n/2+n/2)

print("Length of the parallelised arrays of w:")
print(I_batch[0].shape)


print("Computing the similarity matrix with the parallel loop")
W_vec = similarity_matrix(Fmap, I_batch, J_batch,K,time_adv_mod,geodesic=geodesic)
#results = Parallel(n_jobs=Ncores, verbose = 10)(delayed(similarity_matrix)(Fmap, I_batch[i], J_batch[i],K,time_adv_mod,geodesic=geodesic) for i in range(len(I_batch)))

#Only root saves the data
if rank == root:

    #similarity_matrix now returns the eentire vector, no need for unpacking
    #W_vec = results[0]
    #for res in results[1:]:
    #    W_vec = np.append(W_vec, res)

    #del(results)

    np.save(results_directory+'/'+str(int(tmin/time_steps_per_day))+'_W_matrix_'+W_params+'.npy', W_vec)

    # Create an empty matrix of zeros with shape (n, n)
    W = np.zeros((n, n))
    W[indices] = W_vec
    # Fill the upper triangular part 
    W = W + W.T - np.diag(np.diag(W))
    np.fill_diagonal(W, 0)

    imshow(W)
    cb=plt.colorbar()
    if geodesic==True:
        cb.set_label("m^{-1}")
    else:
        cb.set_label("deg^{-1}")
    plt.title("Similarity matrix")
    plt.savefig(results_directory+'/'+str(int(tmin/time_steps_per_day))+'_W_matrix_'+W_params+'.png')