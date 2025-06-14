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
parser.add_argument("input_files_directory", type=str, help="Directory to the input files")
parser.add_argument("parent_directory", type=str, help="Parent directory")
parser.add_argument("results_directory", type=str, help="Results directory")
parser.add_argument("tmin", type=int, help="Results directory")
# Add optional argument with a default value
parser.add_argument("--K", type=int, default=1000, help="K similarity diagonal (default: 1000)")
# Parse the arguments
args = parser.parse_args()

input_files_directory = args.input_files_directory
parent_directory = args.parent_directory
results_directory = args.results_directory
tmin = args.tmin
K = args.K
time_steps_per_day=4

# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_construction")
sys.path.append(parent_directory+"/subfunctions/latlon_transform")
sys.path.append(parent_directory+"/utils")

#from trajectory_distance import integral_vec
from similarity_matrix import similarity_matrix_mpi       #Commented out since they're currently in this script
from polar_rotation import polar_rotation_rx


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
I_batch = I 
J_batch = J 

print("Number of elements in W triangular:")
print(n*n/2+n/2)

print("Length of the parallelised arrays of w:")
print(I_batch[0].shape)


print("Computing the similarity matrix with the parallel loop")
W_vec = similarity_matrix_mpi(Fmap, I_batch, J_batch,K,time_adv_mod,geodesic=True)


#Only root saves the data
if rank == root:
    np.save(results_directory+'/'+str(int(tmin/time_steps_per_day))+'_W_matrix.npy', W_vec)

    """
    # Create an empty matrix of zeros with shape (n, n)
    W = np.zeros((n, n))
    W[indices] = W_vec
    # Fill the upper triangular part 
    W = W + W.T - np.diag(np.diag(W))
    np.fill_diagonal(W, 0)

    imshow(W)
    cb=plt.colorbar()
    cb.set_label("Km^{-1}")
    
    #if geodesic==True:
        #cb.set_label("Km^{-1}")
    #else:
        #cb.set_label("deg^{-1}")

    plt.title("Similarity matrix")
    plt.savefig(results_directory+'/'+str(int(tmin/time_steps_per_day))+'_W_matrix.png')
    """
