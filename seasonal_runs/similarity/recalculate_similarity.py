#!/usr/bin/env pythonP
# coding: utf-8
import netCDF4 as nc
import sys, os, argparse
import time
import numpy as np
from numpy import ma as ma
from itertools import combinations

#Import packages for plotting
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
from pylab import imshow,cm
import cartopy.crs as ccrs 
import cartopy.feature as cfeature  

#Import packages for clustering
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from scipy.linalg import eigh

#Import packages for geodesic distences
from pyproj import Geod


#Import packages for interpolating and filtering data
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.interpolate import LinearNDInterpolator as LNDI

# Import package for parallel computing
from joblib import Parallel, delayed


import argparse
# Add this at the top of your script
parser = argparse.ArgumentParser(description="Process year and season.")
parser.add_argument("year", type=int, help="Year to process")
parser.add_argument("season", type=str, help="Season to process (e.g., AMJ, OND, JFM, JAS)")
args = parser.parse_args()
# Use the arguments in your script
year = args.year
season = args.season
print(f"Processing year {year}, season {season}")

parent_directory = "/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"

sys.path.append(parent_directory+"/utils")
sys.path.append(parent_directory+"/subfunctions/Parallelisation")
sys.path.append(parent_directory+"/subfunctions/latlon_transform") 
from parallelised_functions import split
from polar_rotation import polar_rotation_rx 



def IC_dist(IC_lat,IC_lon,i_batch,j_batch):
    geod = Geod(ellps='WGS84')  #equivalent to +b=6356752 +a=6378137'
    ic_dist =  []
    for k in range(len(i_batch)):
        if (k%int(len(i_batch)/10) == 0):
            print(k)
        if i_batch[k] == j_batch[k] :  #diagonal elements (same trajectory)
            ic_dist = np.append(ic_dist,1)
        else:
            ic_dist=np.append(ic_dist,geod.inv(IC_lon[0,i_batch[k]],IC_lat[0,i_batch[k]],IC_lon[0,j_batch[k]],IC_lat[0,j_batch[k]])[2]/1000)
    
    #print("number of s = 0 is "+ str(m))
    #print("number of diagonals is "+ str(n))
        
    return ic_dist

IC_resolution = 0.5
dt = 0.0025
DT = 0.1
freq = 1
e = 0
n_clusters = 20
# Format the variables
formatted_e = f"{e:.2f}"
formatted_DT = f"{DT:.4f}"
formatted_dt = f"{dt:.4f}"
# Define other necessary variables
#year = 2009
#season = "AMJ"
# Construct file paths and directories
Fmap_params = f"{year}_{season}_"
Fmap_params += f"ic{IC_resolution}_"
Fmap_params += f"dt{formatted_dt}_"
Fmap_params += f"DT{formatted_DT}"
directory =  f"/cluster/projects/nn8008k/lluisa/NextSIM/seas/" #f"/nird/projects/NS11048K/lluisa/NextSIM/rotated_ice_velocities/seas/AMJ/"
file_path = f"{directory}Fmap/{Fmap_params}/"
parent_directory = "/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"
results_directory = file_path


if not os.path.exists(results_directory):
    os.makedirs(results_directory)

print("Reading data")
#Read input data
Fmap_path = file_path+'/Fmap_matrix.npy'
time_path = file_path+'/advection_time.npy'
W_path = file_path+'/W_matrix.npy'

# Load the Fmap array from the file
Fmap = np.load(Fmap_path)  # ntime [lon,lat] ntrajectories
#Fmap=Fmap[:,:,::100]
# Load the time_adv_mod array from the file
time_adv_mod = np.load(time_path)
# Load the similarity matrix
W_vec = np.load(W_path)


#Initial and final conditions in the standard coordinate system

IC = Fmap[0,:,:]
IC_lat, IC_lon = polar_rotation_rx(IC[1], IC[0],-90)  
FC = Fmap[-1,:,:]
FC_lat, FC_lon = polar_rotation_rx(FC[1], FC[0],-90)  


#Distances between initial conditions

Ncores=32
n=Fmap.shape[2]
indices = np.tril_indices(n,0,n)
I=indices[0]
J=indices[1]


I_batch = list(split(I, Ncores)) # list (Nx*Ny)
J_batch = list(split(J, Ncores)) # list (Nx*Ny)

print("Number of elements in W triangular:")
print(n*n/2+n/2)

print("Length of the parallelised arrays of w:")
print(I_batch[0].shape)

print("Computing distances between initial conditions")
results = Parallel(n_jobs=Ncores, verbose=10)(delayed(IC_dist)(IC_lat, IC_lon, I_batch[i], J_batch[i]) for i in range(len(I_batch)))

ic_dist = results[0]

for res in results[1:]:
    ic_dist = np.append(ic_dist, res)

del(results)



#Distances between final conditions

print("Computing distances between final conditions")
#ic_dist_results = Parallel(n_jobs=Ncores, verbose=10)(delayed(IC_dist)(IC_lat, IC_lon, I_batch[i], J_batch[i]) for i in range(len(I_batch)))
fc_dist_results = Parallel(n_jobs=Ncores, verbose=10)(delayed(IC_dist)(FC_lat, FC_lon, I_batch[i], J_batch[i]) for i in range(len(I_batch)))

#ic_dist = ic_dist_results[0]
fc_dist = fc_dist_results[0]

#for res in ic_dist_results[1:]:
    #ic_dist = np.append(ic_dist, res)

for res in fc_dist_results[1:]:
    fc_dist = np.append(fc_dist, res)

#del(ic_dist_results)
del(fc_dist_results)


w_reweighted = W_vec*ic_dist
w_disp = ic_dist/fc_dist
np.save(file_path+'/W_reweigthed.npy',w_reweighted)
np.save(file_path+'/W_disp.npy',w_disp)