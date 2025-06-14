#!/usr/bin/env python
# coding: utf-8

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
import cartopy.crs as ccrs 
import cartopy.feature as cfeature  

#Import packages for interpolating
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator as LNDI

# Import package for parallel computing
from joblib import Parallel, delayed

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some parameters for clustering.")
# Add required arguments

parser.add_argument("year",type=int, help="Year when the advection is happening")
parser.add_argument("season",type=str, help="season when the advection is happening")

# Parse the arguments
args = parser.parse_args()


year = args.year
season = args.season

#year=2009
#season="AMJ"
# Define variables
Ncores = 32
ndays = 10
# Time step-size (in days)
dt = 0.1 # float
# Spacing of meshgrid (in degrees)
dx = 0.1 # float
dy = 0.1 # float
# Define ratio of auxiliary grid spacing vs original grid_spacing
aux_grid_ratio = .01 # float between [1/100, 1/5]


# Get the parent directory
parent_directory = "/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"
FTLE_parent_directory = "/cluster/home/llpui9007/Programs/FTLE"
TBarrier_parent_directory = "/cluster/home/llpui9007/Programs/TBarrier-main/TBarrier/2D"


formatted_dx = f"{dx:.3f}"
formatted_dy = f"{dy:.3f}"
formatted_dt = f"{dt:.3f}"
formatted_aux_grid_ratio = f"{aux_grid_ratio:.3f}"

# Loop over years and seasons
filename = f"OPA-neXtSIM_CREG025_ILBOXE140_{year}_ice_90Rx_{season}.nc"
directory = "/cluster/projects/nn8008k/lluisa/NextSIM/seas/"
velocities_file_path = os.path.join(directory, filename)
time_steps_per_day=4 #fixed for the NeXtSIM data
dir = "/cluster/projects/nn8008k/lluisa/NextSIM/seas/Fmap_10days/cleaned_FTLE_"+str(year)+"_"+season+"_dx0.100_dy0.100_dt0.100_grid_ratio0.010/"


FTLE_params = f"cleaned_FTLE_"
FTLE_params += f"{year}_{season}_"
FTLE_params += f"dx{formatted_dx}_"
FTLE_params += f"dy{formatted_dy}_"
FTLE_params += f"dt{formatted_dt}_"
FTLE_params += f"grid_ratio{formatted_aux_grid_ratio}"




sys.path.append(parent_directory+"/subfunctions/latlon_transform")
sys.path.append(parent_directory+"/utils")
sys.path.append(FTLE_parent_directory)
# add utils folder to current working path
sys.path.append(TBarrier_parent_directory+"/subfunctions/utils")



# Import linear interpolation function for unsteady flow field with irregular grid
from polar_rotation import polar_rotation_rx 
from ploters import plotpolar_scatter_masked_ftle
from days_since_to_date import days_since_to_date



def plotpolar_scatter_masked_ftle(X_domain, Y_domain, FTLE, mask, t0, t1,  cmap, vmin, vmax,img_name): 
    mask = np.asarray(mask).astype(bool)
    Y_domain_rot, X_domain_rot = polar_rotation_rx(np.array(Y_domain),np.array(X_domain),-90) 

    # Create a figure with a polar stereographic projection centered on the North Pole 
    fig, ax = plt.subplots( 
        figsize=(8, 8), 
        subplot_kw={"projection": ccrs.NorthPolarStereo()}  # North Polar Stereographic projection 
    ) 
    # Choose a colormap (e.g., 'viridis')
    masked = np.array(mask.ravel())
    cax = ax.scatter(np.asarray(X_domain_rot.ravel())[0,~masked], np.asarray(Y_domain_rot.ravel())[0,~masked], c= np.asarray(FTLE.ravel())[~masked], cmap= cmap,transform=ccrs.PlateCarree(), s=8,vmin=vmin,vmax=vmax)
    cbar = fig.colorbar(cax, ticks = np.linspace(0, .4, 9))
    cbar.set_label('FTLE [$\mathrm{days^{-1}}$]', fontsize=18)
    cbar.ax.tick_params(labelsize=14)  # Adjust '14' to your desired tick size
    cbar.set_ticks([vmin, vmin/2, 0, vmax/2, vmax])

    # Add coastlines and gridlines 
    ax.coastlines(resolution='50m', color='black', linewidth=0.8) 
    gl = ax.gridlines(draw_labels=False, color='gray', linestyle='--', linewidth=0.5) 
    gl.ylocator = plt.MultipleLocator(35)  # Latitude gridlines every 35 degrees
    # Set the extent to focus on the North Pole 
    ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree()) 
    # Add a title 
    plt.title(r'$ \mathrm{FTLE}$'+f'$_{{{days_since_to_date(t0)}}}^{{{days_since_to_date(t1)}}}$', fontsize = 24) 
    # Save and show the plot 
    plt.tight_layout() 
    plt.show() 
    plt.savefig(img_name, bbox_inches='tight') 
    plt.close(fig)  # Close the figure to free up memory 



for tmin in range(0,90):
    tmin_idx=tmin*4
    tmax_idx=tmin_idx+10*4
    # Interpolate to a regular grid to then generate the interpolator objects.
    reg_vel_file_path = velocities_file_path[:-3]+'_regrided.nc'


    # Read dataset
    print("Reading regrided input data")
    dataset = nc.Dataset(reg_vel_file_path, mode='r')
    #from m/s to m/day
    interpolated_siu = dataset.variables['vlon'][tmin_idx:tmax_idx,:,:]
    interpolated_siv = dataset.variables['vlat'][tmin_idx:tmax_idx,:,:]
    interpolated_siu = np.transpose(interpolated_siu, axes=(1, 2, 0))
    interpolated_siv = np.transpose(interpolated_siv, axes=(1, 2, 0))
    regrided_land_mask = dataset.variables['land_mask'][:,:]
    # Access coordinates
    lat_grid = dataset.variables['regrided_rot_lat'][:]  
    lon_grid = dataset.variables['regrided_rot_lon'][:]
    # Access specific variables
    time_data = dataset.variables['time'][tmin_idx:tmax_idx] 
    time_data= np.reshape(time_data, (1,-1))
    dataset.close()

    lat_grid = lat_grid.filled()
    lon_grid = lon_grid.filled()

    lat_min, lat_max = lat_grid.min(), lat_grid.max()
    lon_min, lon_max = lon_grid.min(), lon_grid.max()

    x_domain = np.arange(lon_min, lon_max + dx, dx) # array (Nx, )
    y_domain = np.arange(lat_min, lat_max + dy, dy) # array (Ny, )

    X_domain, Y_domain = np.meshgrid(x_domain, y_domain) # array (Ny, Nx)


    ## Compute meshgrid of dataset
    X, Y =  lon_grid, lat_grid # array (NY, NX)

    ## Resolution of meshgrid
    dx_data = X[0,1]-X[0,0] # float
    dy_data = Y[1,0]-Y[0,0] # float

    delta = [dx_data, dy_data] # list (2, )


    # Initial time (in days)
    t0 = time_data[0,0] # float

    # Final time (in days)
    tN = time_data[0,-1] # float

    # NOTE: For computing the backward trajectories: tN < t0 and dt < 0.



    FTLE = np.load(dir+str(tmin)+"_FTLE.npy")
    mask_interpolator = LNDI(list(zip(lat_grid.ravel(), lon_grid.ravel())), regrided_land_mask.ravel(),fill_value=1)
    ftle_land_mask=mask_interpolator(Y_domain,X_domain)
    plotpolar_scatter_masked_ftle(X_domain, Y_domain, FTLE, ftle_land_mask, t0, tN,"seismic",-0.1,0.1, dir+'/'+str(tmin)+"_FTLE.png")


