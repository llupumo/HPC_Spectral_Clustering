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
parser.add_argument("ic_resolution", type=float, help="Lat and lon resolution for the IC ")
parser.add_argument("dt", type=float, help="Time step size for Runge Kutta")
parser.add_argument("DT", type=float, help="Time step size for Fmap")
# Parse the arguments
args = parser.parse_args()

Ncores = args.Ncores
velocities_file_path = args.velocities_file_path
parent_directory = args.parent_directory
results_directory = args.results_directory
ic_resolution = args.ic_resolution
dt = args.dt
DT = args.DT
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
sys.path.append(parent_directory+"/subfunctions/latlon_transform")
sys.path.append(parent_directory+"/utils")

# Import linear interpolation function for unsteady flow field with irregular grid
from Interpolant import interpolant_unsteady
from Interpolant import regrid_unsteady , generate_mask_interpolator , generate_velocity_interpolants
# Import function to compute flow map/particle trajectories
from integration_dFdt import integration_dFdt
from outflow_detector import outflow_detector
from initialize_ic import initialize_ic
from regular_regrid import regular_grid_interpolation, regular_grid_interpolation_scalar
from trajectory_advection import trajectory_advection
#Parallelisation folder
from parallelised_functions import split3D , split 
from NetCDF_generator import save_velocities_to_netCDF, save_velocities_sicsit_to_netCDF, generate_regrided
from polar_rotation import polar_rotation_rx

#########################################################################################

# Interpolate to a regular grid to then generate the interpolator objects.
reg_vel_file_path = velocities_file_path[:-3]+'_regrided.nc'


# Check if the files exist and save them if they don't
if not os.path.exists(reg_vel_file_path):
    print("The velocity has not been regrided yet")
    generate_regrided(reg_vel_file_path,velocities_file_path,latitude_resolution,longitude_resolution,Ncores)

else:
    print("The velocity has allready been regrided. We read the files")
   
# Read dataset
print("Reading the input data")
dataset = nc.Dataset(reg_vel_file_path, mode='r')
#from m/s to m/day
interpolated_siu = dataset.variables['vlon'][:,:,:]
interpolated_siv = dataset.variables['vlat'][:,:,:]
interpolated_siu = np.transpose(interpolated_siu, axes=(1, 2, 0))
interpolated_siv = np.transpose(interpolated_siv, axes=(1, 2, 0))
regrided_land_mask = dataset.variables['land_mask'][:,:]
# Access coordinates
lat_grid = dataset.variables['regrided_rot_lat'][:]  
lon_grid = dataset.variables['regrided_rot_lon'][:]
# Access specific variables
time_data = dataset.variables['time'][:] 
time_data= np.reshape(time_data, (1,-1))
dataset.close()

# Find the points where the velocity arrays are 0. This means either land or null initial velocity and therefore we don't 
# want to have IC there.
vel_land_interpolator = generate_mask_interpolator(lat_grid,lon_grid,interpolated_siu,interpolated_siv)
print("Finished generating the mask interpolator")
Interpolant_u, Interpolant_v = generate_velocity_interpolants(interpolated_siu, interpolated_siv,lon_grid, lat_grid, Ncores)
print("Finished generating the mask interpolator")

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


##########################################################################################################################################################
Fmap, DFDt, time_adv_mod = trajectory_advection(IC, time_data, Interpolant_u, Interpolant_v, lon_grid, lat_grid, timemod, dt, Ncores, melting=True)

np.save(results_directory+'/Fmap_matrix.npy', Fmap)
np.save(results_directory+'/advection_time.npy', time_adv_mod)

#Go back to the non-rotated coordinates to compute the geodesic distances
#Fmap_lat,Fmap_lon = polar_rotation_rx(np.array(Fmap[:,1,:]),np.array(Fmap[:,0,:]),-90)  #Fmap[:,0,:] contains longitudes and Fmap[:,1,:] latitudes
#Fmap_norot = np.stack((np.asarray(Fmap_lon),np.asarray(Fmap_lat)),axis=1)
#np.save(results_directory+'/Fmap_matrix_no_rotated.npy', Fmap_norot)


"""
### Plot some of the advected trajectories
print("Ploting the advected trajectories")

DFDt = DFDt[:-1,:,:]
# Trajectories which velocities vanish to zero
#nulvel_trajectories_idx = np.unique(np.where((abs(DFDt[:-1,0,:]) < 1e-8) & (abs(DFDt[:-1,1,:]) < 1e-8))[1])

velocity_threshold = 0.01
# Check if the velocity is below the threshold for all time steps
condition_x = np.all(abs(DFDt[:, 0, :]) < velocity_threshold, axis=0)
condition_y = np.all(abs(DFDt[:, 1, :]) < velocity_threshold, axis=0)
# Combine conditions for both x and y components
condition = condition_x & condition_y
# Get the indices of trajectories that satisfy the condition
nulvel_trajectories_idx = np.where(condition)[0]

#print("The number of trajectories with constant slow velocity is "+str(len(nulvel_trajectories_idx)))

#Delete trajectories with vanishing velocities
Fmap = np.delete(Fmap,nulvel_trajectories_idx,axis=2)
DFDt = np.delete(DFDt,nulvel_trajectories_idx,axis=2)
"""

















