import sys, os, argparse
import time
import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
from numpy import ma 

#Import packages for plotting
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from pylab import imshow,cm

#Import packages for interpolating
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator as LNDI

# Import package for parallel computing
from joblib import Parallel, delayed


# Create the parser
parser = argparse.ArgumentParser(description="Process some parameters for clustering.")
# Add required arguments
parser.add_argument("Ncores", type=int, help="Number of CPU's")
parser.add_argument("velocities_file_path", type=str, help="Path to the file")
parser.add_argument("parent_directory", type=str, help="Parent directory")
parser.add_argument("results_directory", type=str, help="Results directory")
parser.add_argument("tmin", type=int, help="Minimum time")
parser.add_argument("tmax", type=int, help="Maximum time")
parser.add_argument("ic_resolution", type=float, help="Lat and lon resolution for the IC ")
parser.add_argument("dt", type=float, help="Time step size for Runge Kutta")
parser.add_argument("DT", type=float, help="Time step size for Fmap")
# Add optional argument with a default value
parser.add_argument("--freq", type=int, default=10, help="Frequency (default: 10)")
# Parse the arguments
args = parser.parse_args()

Ncores = args.Ncores
velocities_file_path = args.velocities_file_path
parent_directory = args.parent_directory
results_directory = args.results_directory
tmin = args.tmin
tmax = args.tmax
ic_resolution = args.ic_resolution
dt = args.dt
DT = args.DT
freq = args.freq
timemod = int(DT/dt) 
latitude_resolution = 0.15
longitude_resolution = 0.15


# Create the results directory if it doesn't exist
try:
    os.makedirs(results_directory, exist_ok=True)
    print(f"Directory '{results_directory}' is ready.")
except Exception as e:
    print(f"An error occurred while creating the directory: {e}")
    sys.exit(1)


# add utils folder to current working path
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_clustering")
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_construction")
sys.path.append(parent_directory+"/subfunctions/trajectory_advection")
sys.path.append(parent_directory+"/subfunctions/Parallelisation")
sys.path.append(parent_directory+"/utils")

# Import linear interpolation function for unsteady flow field with irregular grid
from Interpolant import interpolant_unsteady
from Interpolant import regrid_unsteady , generate_mask_interpolator , generate_velocity_interpolants
# Import function to compute flow map/particle trajectories
from integration_dFdt import integration_dFdt
from outflow_detector import outflow_detector
from initialize_ic import initialize_ic
from regular_regrid import regular_grid_interpolation
from trajectory_advection import trajectory_advection
#Parallelisation folder
from parallelised_functions import split3D , split 
from NetCDF_generator import save_velocities_to_netCDF 

#########################################################################################

# Interpolate to a regular grid to then generate the interpolator objects.
reg_vel_file_path = velocities_file_path[:-3]+'_regrided.nc'

# Check if the files exist and save them if they don't

if not os.path.exists(reg_vel_file_path):
    print("The velocity has not been regrided yet")

    # Read dataset
    print("Reading the input data")
    dataset = nc.Dataset(velocities_file_path, mode='r')
    #from m/s to m/day
    siu = dataset.variables['vlon'][:,:,:]
    siv = dataset.variables['vlat'][:,:,:]
    siu = np.transpose(siu, axes=(1, 2, 0))
    siv = np.transpose(siv, axes=(1, 2, 0))
    land_mask=siv[:,:,0].mask
    # Access coordinates
    latitude = dataset.variables['rot_lat'][:]  
    longitude = dataset.variables['rot_lon'][:]
    # Access specific variables
    time_data = dataset.variables['time'][:] 
    netCDF_time_data = time_data
    time_data= np.reshape(time_data, (1,-1))
    dataset.close()

    #### Define a regular grid both for the IC and to use to generate the interpolators
    interpolated_siu, interpolated_siv, lat_grid, lon_grid, regrided_land_mask = regular_grid_interpolation(latitude, longitude, siu, siv,latitude_resolution,longitude_resolution,land_mask,Ncores)
    vel_fillvalue = siu.fill_value
    coord_fillvalue = latitude.fill_value
    del siu
    del siv
    save_velocities_to_netCDF(reg_vel_file_path,netCDF_time_data,interpolated_siu,interpolated_siv,regrided_land_mask, lon_grid,lat_grid,coord_fillvalue,vel_fillvalue)
    print(f"Regrided velocity created.")
    interpolated_siu = interpolated_siu[tmin:tmax,:,:]
    interpolated_siv = interpolated_siv[tmin:tmax,:,:]
    time_data= netCDF_time_data[tmin:tmax] 
    time_data= np.reshape(time_data, (1,-1))

else:
    print("The velocity has allready been regrided. We read the files")
    # Read dataset
    print("Reading the input data")
    dataset = nc.Dataset(reg_vel_file_path, mode='r')
    #from m/s to m/day
    interpolated_siu = dataset.variables['vlon'][tmin:tmax,:,:]
    interpolated_siv = dataset.variables['vlat'][tmin:tmax,:,:]
    interpolated_siu = np.transpose(interpolated_siu, axes=(1, 2, 0))
    interpolated_siv = np.transpose(interpolated_siv, axes=(1, 2, 0))
    regrided_land_mask = dataset.variables['land_mask'][:,:]
    # Access coordinates
    lat_grid = dataset.variables['regrided_rot_lat'][:]  
    lon_grid = dataset.variables['regrided_rot_lon'][:]
    # Access specific variables
    time_data = dataset.variables['time'][tmin:tmax] 
    time_data= np.reshape(time_data, (1,-1))
    dataset.close()

# Find the points where the velocity arrays are 0. This means either land or null initial velocity and therefore we don't 
# want to have IC there.
vel_land_interpolator = generate_mask_interpolator(lat_grid,lon_grid,interpolated_siu,interpolated_siv)
Interpolant_u, Interpolant_v = generate_velocity_interpolants(interpolated_siu, interpolated_siv,lon_grid, lat_grid, Ncores)

del interpolated_siu
del interpolated_siv

lat_min, lat_max = lat_grid.min(), lat_grid.max()
lon_min, lon_max = lon_grid.min(), lon_grid.max()

#### Define initial conditions for advection and keep only sea ice IC (not null velocity, over water)
IC = initialize_ic(lat_min,lat_max,lon_min,lon_max,ic_resolution,vel_land_interpolator)
#Remove conditions in the baltic
IC = outflow_detector(IC,7,30,-68,-40)
#Remove conditions in saint laurens sea
IC = outflow_detector(IC,-40,-35,-70,-60)

# Plot the trajectories
# Create a colormap from blue to red
fig = plt.figure(figsize=(5, 5), dpi=200)
# Define the colors: white for land and light blue for water
ax = plt.axes()
# Plot the land mask
ax.scatter(lon_grid.ravel(), lat_grid.ravel(), marker=".", s=0.01, c=regrided_land_mask)
# Plot initial and final positions
ax.scatter(IC[1,:], IC[0, :], label="IC", c="blue", s=0.5)
# Set axis labels and legend
ax.set_xlim(lon_min - 0.05 * (lon_max - lon_min), lon_max + 0.05 * (lon_max - lon_min))
ax.set_ylim(lat_min - 0.05 * (lat_max - lat_min), lat_max + 0.05 * (lat_max - lat_min))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
# Remove duplicate labels in the legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), fancybox=True)
plt.savefig(results_directory+'/'+str(tmin)+'_IC.png')

##########################################################################################################################################################
Fmap, DFDt, time_adv_mod = trajectory_advection(IC, time_data, Interpolant_u, Interpolant_v, lon_grid, lat_grid, timemod, dt, Ncores)

np.save(results_directory+'/'+str(tmin)+'_Fmap_matrix.npy', Fmap)
np.save(results_directory+'/'+str(tmin)+'_advection_time.npy', time_adv_mod)


### Plot some of the advected trajectories
print("Ploting the advected trajectories")
ntraj_filter = 10
DFDt = DFDt[:-1,:,:]
# Trajectories which velocities vanish to zero
nulvel_trajectories_idx = np.unique(np.where((abs(DFDt[:-1,0,:]) < 1e-8) & (abs(DFDt[:-1,1,:]) < 1e-8))[1])
print("The number of trajectories with vanishing velocity is "+str(len(nulvel_trajectories_idx)))

#plot less trajectories than the ones we have
Fmap_filtered = Fmap[:,:,::ntraj_filter]

#Delete trajectories with vanishing velocities
Fmap = np.delete(Fmap,nulvel_trajectories_idx,axis=2)
DFDt = np.delete(DFDt,nulvel_trajectories_idx,axis=2)
# Plot the trajectories
n, _, m = Fmap_filtered.shape
# Create a colormap from blue to red
cmap = plt.get_cmap('coolwarm', n)
fig = plt.figure(figsize=(5, 5), dpi=200)
# Define the colors: white for land and light blue for water
colors = [(0.58, 0.747, 0.972), (1, 1, 1)]  # Light blue and white
custom_cmap = mcolors.LinearSegmentedColormap.from_list("GreyToBlue", colors)
colors = plt.get_cmap(custom_cmap, 2)

ax = plt.axes()
# Plot the land mask
ax.scatter(lon_grid.ravel(), lat_grid.ravel(), marker=".", s=0.1, c=regrided_land_mask, cmap=colors)
# Plot trajectories with color gradient based on time

for j in range(m):
    # Extract the x and y coordinates for the trajectory
    x = Fmap_filtered[:, 0, j]
    y = Fmap_filtered[:, 1, j]
    
    # Create a color array for the trajectory
    colors = [cmap(i) for i in range(n - 1)]
    
    # Scatter plot for each segment of the trajectory
    ax.scatter(x[:-1], y[:-1], color=colors)

# Plot initial and final positions
ax.scatter(Fmap_filtered[0, 0, :], Fmap_filtered[0, 1, :], label="x(t0)", c="blue", s=5.5)
ax.scatter(Fmap_filtered[-1, 0, :], Fmap_filtered[-1, 1, :], label="x(tN)", c="red", s=5.5)
# Plot vanishing velocity trajectories
#ax.plot(Fmap_filtered[:, 0, nulvel_trajectories_idx_plot], Fmap_filtered[:, 1, nulvel_trajectories_idx_plot], color="yellow", linewidth=1.5, label="Trajectory with vanishing velocity")
# Set axis labels and legend
#ax.set_xlim(lon_min - 0.05 * (lon_max - lon_min), lon_max + 0.05 * (lon_max - lon_min))
#ax.set_ylim(lat_min - 0.05 * (lat_max - lat_min), lat_max + 0.05 * (lat_max - lat_min))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
# Remove duplicate labels in the legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), fancybox=True)
plt.savefig(results_directory+'/'+str(tmin)+'_Advected_trajectories.png')










