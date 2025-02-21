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

# Import package for parallel computing
from joblib import Parallel, delayed

from scipy.interpolate import griddata

# Create the parser
parser = argparse.ArgumentParser(description="Process some parameters for clustering.")
# Add required arguments
parser.add_argument("Ncores", type=int, help="Number of CPU's")
parser.add_argument("input_files_directory", type=str, help="Path to the file")
parser.add_argument("geo_file_path", type=str, help="Path to the velocity field")
parser.add_argument("parent_directory", type=str, help="Parent directory")
parser.add_argument("results_directory", type=str, help="Results directory")
parser.add_argument("geodesic", type=lambda x: x.lower() == 'true', help="Geodesic boolean for trajectory distance")
# Add optional argument with a default value
parser.add_argument("--K", type=int, default=1000, help="K similarity diagonal (default: 1000)")
parser.add_argument("--n_clusters", type=int, default=0, help="Number of clusters (default: 0 which gives the default number)")
parser.add_argument("--e", type=float, default=0, help="Sparsification parameter (default: 0 which translates to standard deviation)")
parser.add_argument("--thereshold", type=float, default=1, help="Thereshold for the radius ")
# Parse the arguments
args = parser.parse_args()


Ncores = args.Ncores
input_files_directory = args.input_files_directory
geo_file_path = args.geo_file_path
parent_directory = args.parent_directory
results_directory = args.results_directory
geodesic = args.geodesic
K = args.K
n_clusters = args.n_clusters
e = args.e
thereshold = args.thereshold
time_steps_per_day = 4
k_exp = 20
tmin = 0
# add utils and subfunctions folders to current working path
sys.path.append(parent_directory+"/subfunctions/border_calculation")
# Import function to compute pairwise distances between trajectories
from ipynb.fs.defs.calculating_borders import calculating_borders
sys.path.append(parent_directory+"/utils")


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


print(int(tmin/time_steps_per_day))

print("Reading data")
#Read input data
dataset = nc.Dataset(geo_file_path, mode='r')

#from m/s to m/day
siu = dataset.variables['vlon'][0,:,:]
siv = dataset.variables['vlat'][0,:,:]
land_mask=siu[:,:].mask
# Access coordinates
latitude = dataset.variables['rot_lat'][:]  
longitude = dataset.variables['rot_lon'][:]
dataset.close()



print("Reading data")
#Read input data

#Read input data
#Fmap_path = input_files_directory+'/'+str(int(tmin/time_steps_per_day))+'_Fmap_matrix.npy'
labels_path = results_directory+str(int(tmin/time_steps_per_day))+'_Clusters_labels_'+Cluster_params+'.npy'
Fmap_path = results_directory+str(int(tmin/time_steps_per_day))+'_Fmap_'+Cluster_params+'.npy'


end_value = 77

# Load the Fmap array from the file
Fmap = np.load(Fmap_path)  # ntime [lon,lat] ntrajectories
# Load the time_adv_mod array from the file
IC = Fmap[0,:,:]  #Take the position of the trajectory IC
# Load the labels of the clusters
labels= np.load(labels_path)
borders=calculating_borders(IC, labels, geodesic, thereshold, Ncores)
# Extract latitude and longitude
longitudes = IC[0, :]
latitudes = IC[1, :]
# Create a grid for interpolation
grid_lon, grid_lat = np.mgrid[min(longitudes):max(longitudes):100j, min(latitudes):max(latitudes):100j]
# Interpolate the data
grid_borders=griddata((longitudes, latitudes), borders, (longitude, latitude), method='cubic')
grid_borders=np.ma.masked_array(grid_borders, land_mask)
# Initialize the vel_land_mask array with False
water_mask = np.full(latitude.shape, False, dtype=bool)
# Compute indices where the velocity is 0 
zero_indices = np.where((siu[:,:] == 0) & (siv[:,:] == 0))
# Set the specified indices to True
water_mask[zero_indices] = True
grid_borders[water_mask]=0
k=1

for tmin in range(4, end_value + 1, 4):
    # Load the Fmap array from the file
    Fmap = np.load(Fmap_path)  # ntime [lon,lat] ntrajectories
    # Load the time_adv_mod array from the file
    IC = Fmap[0,:,:]  #Take the position of the trajectory IC
    # Load the labels of the clusters
    labels= np.load(labels_path)
    borders=calculating_borders(IC, labels, geodesic, thereshold, Ncores)
    # Extract latitude and longitude
    longitudes = IC[0, :]
    latitudes = IC[1, :]
    # Create a grid for interpolation
    grid_lon, grid_lat = np.mgrid[min(longitudes):max(longitudes):100j, min(latitudes):max(latitudes):100j]
    # Interpolate the data
    grid_borders_temp=griddata((longitudes, latitudes), borders, (longitude, latitude), method='cubic')
    grid_borders_temp=np.ma.masked_array(grid_borders_temp, land_mask)
    # Initialize the vel_land_mask array with False
    water_mask_temp = np.full(latitude.shape, False, dtype=bool)
    # Compute indices where the velocity is 0 
    zero_indices = np.where((siu[:,:] == 0) & (siv[:,:] == 0))
    # Set the specified indices to True
    water_mask[zero_indices] = True
    grid_borders_temp[water_mask]=0
    grid_borders = grid_borders+grid_borders_temp
    k+=1

grid_borders = grid_borders/k


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Define color map for the landmask
colors_mask = [(0.58, 0.747, 0.972),
        (1, 1, 1)]  # Grey (RGB for white)  # Light blue (RGB for light sky blue)
# Create the colormap
custom_cmap = mcolors.LinearSegmentedColormap.from_list("GreyToBlue", colors_mask)
colors_mask = plt.get_cmap(custom_cmap, 2)


# Scatter plot
axes[0].pcolormesh(longitude, latitude, land_mask, cmap=custom_cmap, alpha=1)
im = axes[0].scatter(longitudes, latitudes, c=borders, cmap='Oranges', s=8)
axes[0].set_title('Scatter Plot of Initial Conditions')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
fig.colorbar(im, ax=axes[0], label='Borders Value')

# Interpolated plot
#axes[1].scatter(longitude.ravel(), latitude.ravel(),marker='.',s=0.1,c=land_mask.ravel(), cmap=colors_mask)
im = axes[1].pcolor(longitude,latitude,grid_borders,cmap="Blues",alpha=1)
#s=axes[1].pcolormesh(longitude, latitude, land_mask, cmap=colors_mask)
axes[1].set_title('Interpolated Plot of Initial Conditions')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')
fig.colorbar(im, ax=axes[1], label='Borders Value')
# Adjust layout
plt.tight_layout()
# Show plot
plt.show()
plt.savefig(results_directory+"Cluster_Avg_e"+e+"_intdays"+end_value+".png")

#extent=(min(longitudes), max(longitudes), min(latitudes), max(latitudes)),

