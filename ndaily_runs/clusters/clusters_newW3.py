#!/usr/bin/env python
# coding: utf-8

import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import netCDF4 as nc
import sys
import time
import numpy as np
from numpy import ma as ma
from itertools import combinations



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

import argparse
# Add this at the top of your script
parser = argparse.ArgumentParser(description="Process year and season.")
parser.add_argument("year", type=int, help="Year to process")
parser.add_argument("season", type=str, help="Season to process (e.g., AMJ, OND, JFM, JAS)")
parser.add_argument("tmin", type=int, help="Minimum time")
args = parser.parse_args()
# Use the arguments in your script
year = args.year
season = args.season
tmin = args.tmin*4
print(f"Processing year {year}, season {season}")

IC_resolution = 0.5
dt = 0.0025
DT = 0.01
freq = 1
e = 0
#n_clusters = 20
# Format the variables
formatted_e = f"{e:.2f}"
formatted_DT = f"{DT:.4f}"
formatted_dt = f"{dt:.4f}"

# Construct file paths and directories
Fmap_params = f"{year}_{season}_"
Fmap_params += f"ic{IC_resolution}_"
Fmap_params += f"dt{formatted_dt}_"
Fmap_params += f"DT{formatted_DT}"
directory =  f"/cluster/projects/nn8008k/lluisa/NextSIM/seas/" #f"/nird/projects/NS11048K/lluisa/NextSIM/rotated_ice_velocities/seas/AMJ/"
file_path = f"{directory}Fmap_10days/{Fmap_params}/"
parent_directory = "/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"
results_directory = file_path
regrided_geo_file_path = f"{directory}OPA-neXtSIM_CREG025_ILBOXE140_{year}_ice_90Rx_{season}_regrided.nc"
geo_file_path = f"{directory}OPA-neXtSIM_CREG025_ILBOXE140_{year}_ice_90Rx_{season}.nc"
distance = 0 #0
k_exp = 100
d=10000 #e=0
d_reweighted=10000 #e=0
d_disp=10000 #e=0
time_steps_per_day=4


if not os.path.exists(results_directory):
    os.makedirs(results_directory)


# add utils folder to the TBarrier package
#sys.path.append(T_Barrier_directory+"/subfunctions/utils")
#sys.path.append(T_Barrier_directory+"/subfunctions/integration")
# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_clustering")
sys.path.append(parent_directory+"/utils")
sys.path.append(parent_directory+"/subfunctions/Parallelisation")
sys.path.append(parent_directory+"/subfunctions/latlon_transform") 
from parallelised_functions import split3D

sys.path.append(parent_directory+"/subfunctions/trajectory_advection")
from Interpolant import generate_land_mask_interpolator 

from from_similarity_to_eigen import from_similarity_to_eigen, from_similarity_to_eigen_W, cut_trajectories_in_W , cut_trajectories_in_3W, from_similarity_to_eigen_W_spard #, from_similarity_to_eigesut_zones

from ploters import ini_final_clusters
from ploters import gif_clusters
from ploters import ini_final_clusters_landmask
from ploters import ini_final_clusters_landmask_ini
from ploters import gif_clusters_landmask
from degrees import degree_matrix
from polar_rotation import polar_rotation_rx 


# ### Clustering
def kmeans_w(n_clusters,l_vect):
    print("Applying k-means to define the clusters")

    kmeans = KMeans(init="random",n_clusters=n_clusters,n_init=1000,max_iter=10000)
    kmeans.fit(l_vect[:,0:n_clusters])
    labels = kmeans.labels_

    #plot_clusters_3w(Fmap, n_clusters, labels_reweighted, labels_disp, labels, "tab20")
    return labels



def kmeans_3w(n_clusters,l_vect_reweighted,l_vect_disp,l_vect,Fmap,d):
    print("Applying k-means to define the clusters")

    kmeans = KMeans(init="random",n_clusters=n_clusters,n_init=1000,max_iter=10000)
    kmeans.fit(l_vect[:,0:n_clusters])
    labels = kmeans.labels_
    kmeans.fit(l_vect_reweighted[:,0:n_clusters])
    labels_reweighted = kmeans.labels_
    kmeans.fit(l_vect_disp[:,0:n_clusters])
    labels_disp = kmeans.labels_


    Cluster_params = (
        f"nclusters{n_clusters}_"
        f"d{d:.2f}"
    )

    #plot_clusters_3w(Fmap, n_clusters, labels_reweighted, labels_disp, labels, "tab20")
    return labels_reweighted, labels_disp, labels



print("Reading data")
#Read input data
Fmap_path = file_path+'/'+str(int(tmin/time_steps_per_day))+'_Fmap_matrix_cleaned.npy'
time_path = file_path+'/'+str(int(tmin/time_steps_per_day))+'_advection_time.npy'
W_path = file_path+'/'+str(int(tmin/time_steps_per_day))+'_W_matrix_cleaned.npy'
W_path_reweighted = file_path+'/'+str(int(tmin/time_steps_per_day))+'_W_reweighted_cleaned.npy'
W_path_disp = file_path+'/'+str(int(tmin/time_steps_per_day))+'_W_disp_cleaned.npy'

# Load the Fmap array from the file
Fmap = np.load(Fmap_path)  # ntime [lon,lat] ntrajectories
#Fmap=Fmap[:,:,::100]
# Load the time_adv_mod array from the file
time_adv_mod = np.load(time_path)
# Load the similarity matrix
W_vec = np.load(W_path)
w_disp = np.load(W_path_disp)
w_reweighted = np.load(W_path_reweighted)

dataset = nc.Dataset(regrided_geo_file_path, mode='r')
land_mask_reg = dataset.variables['land_mask'][:,:]
# Access coordinates
latitude_reg = dataset.variables['regrided_rot_lat'][:]  
longitude_reg = dataset.variables['regrided_rot_lon'][:]
dataset.close()


print("Cutting the trajectories that we don't want")
#W, Fmap_cut = cut_trajectories_in_W(Fmap, W_vec, distance,land_mask_reg,latitude_reg,longitude_reg)
#W_disp, Fmap_cut = cut_trajectories_in_W(Fmap, w_disp, distance,land_mask_reg,latitude_reg,longitude_reg)
#W_reweighted, Fmap_cut = cut_trajectories_in_W(Fmap, w_reweighted, distance,land_mask_reg,latitude_reg,longitude_reg)

W_disp, Fmap_cut = cut_trajectories_in_W(Fmap, w_disp, distance,land_mask_reg,latitude_reg,longitude_reg, eastSval=True, canadian_greenland=True)

K = 1000
np.fill_diagonal(W_disp,K)

formatted_distance = f"{distance:.2f}"

clusters_path = file_path+'/disp_clusters_K'+str(K)+'_border'+str(formatted_distance)+'_cleaned_center_spars/'
if not os.path.exists(clusters_path):
    os.makedirs(clusters_path)


spar = 10000
print("Sparsification distance: ")
print(str(spar))
if spar==10000:
    z=0
    formatted_spar = f"{z:.2f}"
else:
    formatted_spar = f"{spar:.2f}"

l_vect,l,Fmap_cut_dist = from_similarity_to_eigen_W_spard(Fmap_cut,spar,W_disp,W_disp,K,k_exp)
if not os.path.exists(clusters_path):
    os.makedirs(clusters_path)

np.save(clusters_path+'tmin'+str(int(tmin/time_steps_per_day))+'_spar'+str(formatted_spar)+'_labels'+'_Fmap_cut.npy',Fmap_cut_dist)
np.save(clusters_path+'tmin'+str(int(tmin/time_steps_per_day))+'_spar'+str(formatted_spar)+'_eigenvalues.npy',l)




print("Clustering")
#for n_clusters in (15,20, 25,30, 35):
#n_clusters = 15
for n_clusters in range(2,100,1):
    #if not os.path.exists(clusters_path+str(n_clusters)+'_Fmap_cut.npy'):  
    labels = kmeans_w(n_clusters,l_vect)

    print("Saving results")
    np.save(clusters_path+str(n_clusters)+'_tmin'+str(int(tmin/time_steps_per_day))+'_spar'+str(formatted_spar)+'_labels.npy',labels)
    