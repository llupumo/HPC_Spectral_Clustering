#!/usr/bin/env python
# coding: utf-8

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


IC_resolution = 0.5
dt = 0.0025
DT = 0.1
freq = 1
geodesic = True
e = 0
n_clusters = 20
# Format the variables
formatted_e = f"{e:.2f}"
formatted_DT = f"{DT:.4f}"
formatted_dt = f"{dt:.4f}"
# Define other necessary variables
year = 2009
season = "AMJ"
# Construct file paths and directories
Fmap_params = f"{year}_{season}_"
Fmap_params += f"ic{IC_resolution}_"
Fmap_params += f"dt{formatted_dt}_"
Fmap_params += f"DT{formatted_DT}"
directory = f"/cluster/projects/nn8008k/lluisa/NextSIM/rotated_ice_velocities/seas/{season}/"
file_path = f"{directory}Fmap/{Fmap_params}/"
parent_directory = "/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"
results_directory = file_path
regrided_geo_file_path = f"{directory}OPA-neXtSIM_CREG025_ILBOXE140_{year}_ice_90Rx_{season}_regrided.nc"
geo_file_path = f"{directory}OPA-neXtSIM_CREG025_ILBOXE140_{year}_ice_90Rx_{season}.nc"
K=1000
distance = 4
k_exp = 100

if not os.path.exists(results_directory):
    os.makedirs(results_directory)


Cluster_params = (
    f"geodesic_{geodesic}_"
    f"nclusters{n_clusters}_"
    f"e{e:.2f}"
)

W_params = (
    f"geodesic_{geodesic}"
)



# add utils folder to the TBarrier package
#sys.path.append(T_Barrier_directory+"/subfunctions/utils")
#sys.path.append(T_Barrier_directory+"/subfunctions/integration")
# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_clustering")
sys.path.append(parent_directory+"/utils")
sys.path.append(parent_directory+"/subfunctions/Parallelisation")
from parallelised_functions import split3D

sys.path.append(parent_directory+"/subfunctions/trajectory_advection")
from Interpolant import generate_land_mask_interpolator 

from from_similarity_to_eigen import from_similarity_to_eigen, from_similarity_to_eigen_W, cut_trajectories_in_W

from ploters import ini_final_clusters
from ploters import gif_clusters
from ploters import ini_final_clusters_landmask
from ploters import ini_final_clusters_landmask_ini
from ploters import gif_clusters_landmask
from degrees import degree_matrix


print("Reading data")
#Read input data
Fmap_path = file_path+'/Fmap_matrix.npy'
time_path = file_path+'/advection_time.npy'
W_path = file_path+'/W_matrix_'+W_params+'.npy'

# Load the Fmap array from the file
Fmap = np.load(Fmap_path)  # ntime [lon,lat] ntrajectories
#Fmap=Fmap[:,:,::100]
# Load the time_adv_mod array from the file
time_adv_mod = np.load(time_path)
# Load the similarity matrix
W_vec = np.load(W_path)

dataset = nc.Dataset(regrided_geo_file_path, mode='r')
#from m/s to m/day
siu = dataset.variables['vlon'][0,:,:]
land_mask_reg = dataset.variables['land_mask'][:,:]
# Access coordinates
latitude_reg = dataset.variables['regrided_rot_lat'][:]  
longitude_reg = dataset.variables['regrided_rot_lon'][:]
dataset.close()

dataset = nc.Dataset(geo_file_path, mode='r')
#from m/s to m/day
land_mask = dataset.variables['vlon'][0,:,:].mask
print("shape of land mask")
print(str(land_mask.shape))
# Access coordinates
latitude = dataset.variables['rot_lat'][:]  
longitude = dataset.variables['rot_lon'][:]
dataset.close()

print("Computing the eigenvalues")
if e==0:
    e = np.std(W_vec[W_vec<999])


fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(W_vec[W_vec<999], color = 'black', label="No_spars")
axes[0].axhline(y=e, color='yellow', linestyle='--', label=f'e = {e}')
axes[0].legend()


W, Fmap_cut = cut_trajectories_in_W(Fmap, W_vec,e,distance,land_mask_reg,latitude_reg,longitude_reg)
l_vect,l,Fmap,n_clusters_def = from_similarity_to_eigen_W(Fmap_cut,W,K,k_exp)
#l_vect,l,Fmap,n_clusters_def = from_similarity_to_eigen_cut_zones(Fmap, W_vec, e, K, k_exp, distance,land_mask_reg,latitude_reg,longitude_reg)

lx = np.arange(1, len(l) + 1)
axes[1].plot(lx,l,marker='.',color='red',label=str(k_exp)+" first eigenvalues")
axes[1].axvline(x=n_clusters_def, color="green",label="default number of clusters")
axes[1].legend()
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].set_xlabel("Index")
axes[1].set_ylabel("Generalised eigenvalues")
plt.show
plt.savefig(results_directory+"cut0_eigenvalues_"+Cluster_params+".png")


"""
def cut_weights(W,Fmap,labels):
    
    #This function calculates the cost of the cut of each cluster with respect to the other clusters. It returns a vector of length number of initial conditions 
    with the between cluster cut cost of the cut
"""
    
n_clusters = 9

Cluster_params = (
    f"geodesic_{geodesic}_"
    f"nclusters{n_clusters}_"
    f"e{e:.2f}"
)

# ### Clustering
print("Applying k-means to define the clusters")
if n_clusters==0:
    n_clusters = n_clusters_def  
l_vect_cut = l_vect[:,0:n_clusters]
kmeans = KMeans(init="random",n_clusters=n_clusters,n_init=1000,max_iter=10000)
kmeans.fit(l_vect_cut)
labels = kmeans.labels_



np.save(results_directory+'/cut4_Clusters_labels_'+Cluster_params+'.npy', labels)
np.save(results_directory+'/cut4_Fmap_'+Cluster_params+'.npy', Fmap)

print("Plotting the clusters")

"""
for n_clusters in range(2,10):
    Cluster_params = (
        f"geodesic_{geodesic}_"
        f"nclusters{n_clusters}_"
        f"e{e:.2f}"
    )

    # ### Clustering
    print("Applying k-means to define the clusters")
    if n_clusters==0:
        n_clusters = n_clusters_def  
    l_vect_cut = l_vect[:,0:n_clusters]
    kmeans = KMeans(init="random",n_clusters=n_clusters,n_init=1000,max_iter=10000)
    kmeans.fit(l_vect_cut)
    labels = kmeans.labels_

    np.save(results_directory+'/cut4_Clusters_labels_'+Cluster_params+'.npy', labels)
    np.save(results_directory+'/cut4_Fmap_'+Cluster_params+'.npy', Fmap)


    print("Plotting the clusters")
    #ini_final_clusters(Fmap, n_clusters, labels, results_directory, "", e)
    #ini_final_clusters_landmask(Fmap, n_clusters, labels, results_directory+"clusters"+Cluster_params+"_0.png", e, longitude, latitude, land_mask)
    #ini_final_clusters_landmask_ini(Fmap, n_clusters, labels, results_directory+"cut0_clusters"+Cluster_params+"_ini.png", e, longitude, latitude, land_mask)

"""