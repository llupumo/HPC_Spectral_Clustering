#####################to be fixed!!

#!/usr/bin/env pythonP
# coding: utf-8

import netCDF4 as nc
import sys, os, argparse
import numpy as np
from numpy import ma as ma

#Import packages for plotting
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from pylab import imshow,cm
#Import packages for plotting
from matplotlib.colors import ListedColormap
from pylab import imshow,cm
import cartopy.crs as ccrs 
import cartopy.feature as cfeature  

#Import packages for clustering
from sklearn.cluster import KMeans
from scipy.linalg import eigh

#Import packages for geodesic distences
from pyproj import Geod

# Import package for parallel computing
from joblib import Parallel, delayed

from scipy.interpolate import griddata


import argparse
# Add this at the top of your script
parser = argparse.ArgumentParser(description="Process year and season.")
parser.add_argument("year", type=int, help="Year to process")
parser.add_argument("season", type=str, help="Season to process (e.g., AMJ, OND, JFM, JAS)")
parser.add_argument("tmin", type=int, help="start time step")
args = parser.parse_args()
# Use the arguments in your script
year = args.year
season = args.season
tmin = args.tmin
time_steps_per_day=4
print(f"Processing year {year}, season {season}")


IC_resolution = 0.5
dt = 0.0025
DT = 0.1
# Format the variables
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
regrided_geo_file_path = f"{directory}OPA-neXtSIM_CREG025_ILBOXE140_{year}_ice_90Rx_{season}_regrided.nc"

NCores=32
K=10**7
distance = 0

formatted_distance = f"{distance:.2f}"

clusters_path =  "/cluster/projects/nn8008k/lluisa/NextSIM/seas/Fmap_10days/"+str(year)+"_"+season+"_ic0.5_dt0.0025_DT0.0100/disp_clusters_K1000_border0.00_cleaned_center_spars/" 



# Get the parent directory
parent_directory = "/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"
FTLE_parent_directory = "/cluster/home/llpui9007/Programs/FTLE"
TBarrier_parent_directory = "/cluster/home/llpui9007/Programs/TBarrier-main/TBarrier/2D"


# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/trajectory_advection")
sys.path.append(parent_directory+"/subfunctions/latlon_transform")
sys.path.append(parent_directory+"/utils")
sys.path.append(FTLE_parent_directory)
# add utils folder to current working path
sys.path.append(TBarrier_parent_directory+"/subfunctions/utils")
sys.path.append(TBarrier_parent_directory+"/subfunctions/integration")
# add FTLE folder to current working path
sys.path.append(TBarrier_parent_directory+"/demos/AdvectiveBarriers/FTLE2D")


# add utils folder to the TBarrier package
#sys.path.append(T_Barrier_directory+"/subfunctions/utils")
#sys.path.append(T_Barrier_directory+"/subfunctions/integration")
# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_clustering")
sys.path.append(parent_directory+"/subfunctions/border_calculation")
sys.path.append(parent_directory+"/utils")
sys.path.append(parent_directory+"/subfunctions/Parallelisation")
from parallelised_functions import split

# Import linear interpolation function for unsteady flow field with irregular grid
from Interpolant import generate_mask_interpolator , generate_velocity_interpolants, interpolant_unsteady_FTLE
# Import function to compute flow map/particle trajectories
from regular_regrid import regular_grid_interpolation_scalar

# Import function to compute finite time Lyapunov exponent (FTLE)
from FTLE import parallel_FTLE
from polar_rotation import polar_rotation_rx 
from calculating_borders import borders_binary

ftle_path = "/cluster/projects/nn8008k/lluisa/NextSIM/seas/Fmap_10days/cleaned_FTLE_"+str(year)+"_"+season+"_dx0.100_dy0.100_dt0.100_grid_ratio0.010/"
domain_path = "/cluster/projects/nn8008k/lluisa/NextSIM/seas/Fmap_10days/"
X_domain = np.load(domain_path+"X_domain.npy")
Y_domain = np.load(domain_path+"Y_domain.npy")
FTLE_field = np.load(ftle_path + str(tmin) + "_FTLE.npy")
Interpolant_FTLE = interpolant_unsteady_FTLE(X_domain, Y_domain, FTLE_field)

print("Reading regrided input data")
dataset = nc.Dataset(regrided_geo_file_path, mode='r')
# Access coordinates
lat_grid = dataset.variables['regrided_rot_lat'][:]  
lon_grid = dataset.variables['regrided_rot_lon'][:]
dataset.close()

formatted_d=str(0)
spar=0
formatted_spar = f"{spar:.2f}"
FTLE = []



for n_clusters in range (2, 54): #:100):
    print("Processing "+str(n_clusters)+" clusters")
    Fmap_path = clusters_path+'tmin'+str(int(tmin))+'_spar'+str(formatted_spar)+'_labels'+'_Fmap_cut.npy'
    labels_path = clusters_path+str(n_clusters)+'_tmin'+str(int(tmin))+'_spar'+str(formatted_spar)+'_labels.npy'
    labels_disp_path = clusters_path+str(n_clusters)+'_tmin'+str(int(tmin))+'_spar'+str(formatted_spar)+'_labels_disp.npy'
    labels_reweighted_path = clusters_path+str(n_clusters)+'_tmin'+str(int(tmin))+'_spar'+str(formatted_spar)+'_labels_reweighted.npy'

    Fmap = np.load(Fmap_path)
    labels= np.load(labels_path)
    labels_disp = np.load(labels_disp_path)
    labels_reweighted = np.load(labels_reweighted_path)

    IC = Fmap[0,:,:]  #Take the position of the trajectory IC
    # Load the labels of the clusters
    grid_labels = griddata((IC[0, :], IC[1, :]), labels_disp, (lon_grid,lat_grid), method='nearest')
    fmap_mask = np.isnan(griddata((IC[0, :], IC[1, :]), labels_disp, (lon_grid,lat_grid), method='linear'))
    borders_avg = borders_binary(grid_labels)
    borders_avg = np.where(fmap_mask,np.nan, borders_avg)
    borders_idx = np.where(borders_avg==1)
    borders_lon_rot = lon_grid[borders_idx]
    borders_lat_rot = lat_grid[borders_idx]
    FTLE.append(Interpolant_FTLE(borders_lat_rot, borders_lon_rot, grid=False))


mean_FTLE = []
idx = []
for i in range(0,len(FTLE)):
    mean_FTLE.append(np.mean(FTLE[i]))
    idx.append(2+i)

plt.plot(idx,mean_FTLE)

np.save(clusters_path+"/2_100_FTLE_interpol_"+str(tmin)+".npy", FTLE)
print("ftle.npy saved")
plt.savefig(clusters_path+"/2_100_FTLE_interpol_"+str(tmin)+".png")





