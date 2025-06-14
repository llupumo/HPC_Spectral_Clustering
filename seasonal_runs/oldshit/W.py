import netCDF4 as nc
import sys, os, argparse
import time
import numpy as np
from numpy import ma as ma

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
# Add optional argument with a default value
parser.add_argument("--K", type=int, default=1000, help="K similarity diagonal (default: 1000)")
# Parse the arguments
args = parser.parse_args()

Ncores = args.Ncores
input_files_directory = args.input_files_directory
parent_directory = args.parent_directory
results_directory = args.results_directory
geodesic = args.geodesic
K = args.K


W_params = (
    f"geodesic_{geodesic}"
)

# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_construction")
sys.path.append(parent_directory+"/subfunctions/latlon_transform")
sys.path.append(parent_directory+"/utils")

from similarity_matrix import similarity_matrix
from trajectory_distance import integral_vec

####################################################################################

#Read input data
Fmap_path = input_files_directory+'/Fmap_matrix.npy'
time_path = input_files_directory+'/advection_time.npy'
# Load the Fmap array from the file
Fmap = np.load(Fmap_path)  # ntime [lon,lat] ntrajectories
# Load the time_adv_mod array from the file
time_adv_mod = np.load(time_path)



print("Preparing the parallel loop to compute the Similarity matrix") 
n = Fmap.shape[2]
print(f"{n} trajectories are being processed. Each trajectory has {Fmap.shape[0]} time steps.")

indices = np.tril_indices(n,0,n)

I=indices[0]
J=indices[1]

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

I_batch = list(split(I, Ncores)) # list (Nx*Ny)
J_batch = list(split(J, Ncores)) # list (Nx*Ny)

print("Number of elements in W triangular:")
print(n*n/2+n/2)

print("Length of the parallelised arrays of w:")
print(I_batch[0].shape)


print("Computing the similarity matrix with the parallel loop")
results = Parallel(n_jobs=Ncores, verbose = 10)(delayed(similarity_matrix)(Fmap, I_batch[i], J_batch[i],K,time_adv_mod,geodesic=geodesic) for i in range(len(I_batch)))

W_vec = results[0]

for res in results[1:]:
    W_vec = np.append(W_vec, res)

del(results)

np.save(results_directory+'/W_matrix_'+W_params+'.npy', W_vec)

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
plt.savefig(results_directory+'/W_matrix_'+W_params+'.png')

