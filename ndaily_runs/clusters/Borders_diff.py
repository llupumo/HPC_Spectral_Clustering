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
#Import packages for plotting
from matplotlib.colors import ListedColormap

#Import packages for clustering
from sklearn.cluster import KMeans
from scipy.linalg import eigh

#Import packages for geodesic distences
from pyproj import Geod

# Import package for parallel computing
from joblib import Parallel, delayed

from scipy.interpolate import griddata

import cartopy.crs as ccrs 
import cartopy.feature as cfeature  

parent_directory = "/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering/"


# add utils folder to the TBarrier package
#sys.path.append(T_Barrier_directory+"/subfunctions/utils")
#sys.path.append(T_Barrier_directory+"/subfunctions/integration")
# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_clustering")
sys.path.append(parent_directory+"/subfunctions/border_calculation")
sys.path.append(parent_directory+"/utils")


from ploters import ini_final_clusters
from ploters import gif_clusters
from ploters import ini_final_clusters_landmask
from ploters import gif_clusters_landmask
from calculating_borders import borders_binary

#Import packages for geodesic distences 
from pyproj import Geod

# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_construction")
sys.path.append(parent_directory+"/subfunctions/latlon_transform")
sys.path.append(parent_directory+"/utils")

#from trajectory_distance import integral_vec
#from similarity_matrix import similarity_matrix        #Commented out since they're currently in this script
from polar_rotation import polar_rotation_rx


IC_resolution = 0.5
dt = 0.0025
DT = 0.01
freq = 1
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
directory =  f"/cluster/projects/nn8008k/lluisa/NextSIM/seas/" #f"/nird/projects/NS11048K/lluisa/NextSIM/rotated_ice_velocities/seas/AMJ/"
file_path = f"{directory}Fmap_10days/{Fmap_params}/"
parent_directory = "/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"
results_directory = file_path
regrided_geo_file_path = f"{directory}OPA-neXtSIM_CREG025_ILBOXE140_{year}_ice_90Rx_{season}_regrided.nc"
geo_file_path = f"{directory}OPA-neXtSIM_CREG025_ILBOXE140_{year}_ice_90Rx_{season}.nc"
K=1000
distance = 4
k_exp = 100


print("reading")
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


# Define the 30 qualitative colors
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194"
]
# Create a ListedColormap
cmap = ListedColormap(colors, name='qualitative_30')
def plot_clusters_3w(Fmap, n_clusters, labels_reweighted, labels_disp, labels, cmap, img_name, tmin): 
    # Define a diverging colormap and normalization centered at 1 with an asymmetric range
    #vlim_reweighted = max(abs(1-w_reweighted.min()),abs(1-w_reweighted[w_reweighted<w_reweighted.max()].max()))
    #vlim_disp = max(abs(1-w_disp.min()),abs(1-w_disp[w_disp<w_disp.max()].max()))
    IC = Fmap[0,:,:]
    IC_lat, IC_lon = polar_rotation_rx(IC[1], IC[0],-90) 
    positions_ini = np.asarray(np.vstack((IC_lon,IC_lat))) 

    # Create a figure with two subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,  # 1 row, 2 columns
        figsize=(24, 8),  # Adjust the figure size
        subplot_kw={"projection": ccrs.NorthPolarStereo()}  # North Polar Stereographic projection
    )

    # Define a color map for the clusters
    colors = plt.get_cmap(cmap, n_clusters)
    cluster_names = [f"Cluster {i+1}" for i in range(n_clusters)]  # Names for legend

    # Plot the first subplot with w_reweighted
    # Plot the initial distribution
    ax1.scatter(positions_ini[0, :], positions_ini[1, :], c=labels_reweighted, cmap=colors, vmin=0, vmax=n_clusters-1,transform=ccrs.PlateCarree(), s=8)
    ax1.coastlines(resolution='50m', color='black', linewidth=0.8)
    gl1 = ax1.gridlines(draw_labels=False, color='gray', linestyle='--', linewidth=0.5)
    gl1.xlocator = plt.MultipleLocator(45)  # Longitude gridlines every 45 degrees
    gl1.ylocator = plt.MultipleLocator(35)  # Latitude gridlines every 35 degrees
    ax1.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
    ax1.set_title("Re-weighted Lagrangian distance", fontsize=20)
  
    # Plot the second subplot with w_disp
    ax2.scatter(positions_ini[0, :], positions_ini[1, :], c=labels_disp, cmap=colors, vmin=0, vmax=n_clusters-1,transform=ccrs.PlateCarree(), s=8)
    ax2.coastlines(resolution='50m', color='black', linewidth=0.8)
    gl1 = ax2.gridlines(draw_labels=False, color='gray', linestyle='--', linewidth=0.5)
    gl1.xlocator = plt.MultipleLocator(45)  # Longitude gridlines every 45 degrees
    gl1.ylocator = plt.MultipleLocator(35)  # Latitude gridlines every 35 degrees
    ax2.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
    ax2.set_title("Dispersion measure", fontsize=20)

    # Plot the second subplot with w_vec
    ax3.scatter(positions_ini[0, :], positions_ini[1, :], c=labels, cmap=colors, vmin=0, vmax=n_clusters-1,transform=ccrs.PlateCarree(), s=8)
    ax3.coastlines(resolution='50m', color='black', linewidth=0.8)
    gl2 = ax3.gridlines(draw_labels=False, color='gray', linestyle='--', linewidth=0.5)
    gl2.xlocator = plt.MultipleLocator(45)  # Longitude gridlines every 45 degrees
    gl2.ylocator = plt.MultipleLocator(35)  # Latitude gridlines every 35 degrees
    ax3.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
    ax3.set_title("Lagrangian distance ", fontsize=20)

    # Adjust layout and show the plot
    fig.suptitle(str(n_clusters)+" clusters,   tmin: "+str(tmin),fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.savefig(img_name, bbox_inches='tight')  # Save the figure if needed
    plt.close(fig)  # Close the figure to free up memory

n_clusters_list = [15,30]
for tmin in range(0,81):
    print(tmin)
    for i in n_clusters_list:
        print(i)
        clusters_dir="/cluster/projects/nn8008k/lluisa/NextSIM/seas/Fmap_10days/2009_AMJ_ic0.5_dt0.0025_DT0.0100/clusters_K10000000_border4.00/"
        labels_path = clusters_dir+str(i)+"_tmin"+str(tmin)+"_d0_labels.npy"
        labels=np.load(labels_path)
        labels_disp_path = clusters_dir+str(i)+"_tmin"+str(tmin)+"_d0_labels_disp.npy"
        labels_disp=np.load(labels_disp_path)
        labels_reweighted_path = clusters_dir+str(i)+"_tmin"+str(tmin)+"_d0_labels_reweighted.npy"
        labels_reweighted=np.load(labels_reweighted_path)
        Fmap_path = clusters_dir+str(i)+"_tmin"+str(tmin)+"_Fmap_cut.npy"
        Fmap=np.load(Fmap_path)
        plot_clusters_3w(Fmap, i, labels_reweighted, labels_disp, labels, cmap,clusters_dir+str(i)+"clusters_tmin"+str(tmin)+".png",tmin)