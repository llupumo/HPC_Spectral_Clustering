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

seasons = "JFM OND AMJ JAS"
# Split the string into a list of seasons
season_list = seasons.split()
# Loop through each season and print it

def ini_final_clusters_landmask_borders_seas(Fmap, borders_avg, img_name, e, x, y, mask_interpol, scalar_ini):
    positions_ini = Fmap[0, :, :]
    positions_end = Fmap[-1, :, :]

    ymax = y.max()
    ymin = y.min()
    xmax = x.max()
    xmin = x.min()

    fig, ax = plt.subplots(figsize=(8, 6))
    # Define a color map for the clusters
    

    # First subplot
    #scalar_ini = scalar[-1, :, :]  # Assuming you want to show the first time slice
    #ax.pcolor(x,y,fmap_mask,cmap="Greys",alpha=1)
    im1 = ax.pcolormesh(x,y,borders_avg,cmap="jet",alpha=1)
    ax.imshow(scalar_ini, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='bone')
    
    #scatter_ini = ax.scatter(positions_ini[0, :], positions_ini[1, :], c=labels, cmap=colors, vmin=0, vmax=n_clusters-1, s=4)
    contour_land = ax.contour(x, y, mask_interpol, levels=0, cmap='bone', alpha=1)

    #im1 = ax.imshow(masked_borders_ini, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='bone_r', alpha=0.7)

    # Add contour plot to the first subplot
    #contour_land = ax.contour(x, y, mask_interpol, levels=1, cmap='viridis', alpha=0.5)
    ax.set_xlabel("Rotated Longitude")
    ax.set_ylabel("Rotated Latitude")
    ax.set_title("Initial distribution of the clusters")
    ax.set_xlim(xmin-0.05*(xmax-xmin), xmax+0.05*(xmax-xmin)) 
    ax.set_ylim(ymin-0.05*(xmax-xmin), ymax+0.05*(ymax-ymin))  
    ax.set_aspect('equal', 'box')

    # Add legend

    #axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))
    cbar = fig.colorbar(im1, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    #cbar.set_label('Colorbar Label')  # Optional: add a label to the colorbar
    # Main title
    plt.suptitle(f"{e} spars", fontsize=16)
    plt.subplots_adjust(right=0.8) 
    # Save the figure
    plt.savefig(f"{img_name}")
    # Show the plot
    plt.show()

for season in season_list:

    year = "2009"
    geo_file_path = "/cluster/projects/nn9970k/lluisa/NextSIM/rotated_ice_velocities/seas/"+season+"/OPA-neXtSIM_CREG025_ILBOXE140_"+year+"_ice_90Rx_"+season+"_regrided.nc"
    file_path="/cluster/projects/nn9970k/lluisa/NextSIM/rotated_ice_velocities/seas/"+season+"/Fmap/"+year+"_"+season+"_ic0.5_dt0.0025_DT0.1000/"


    parent_directory = "/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering/"
    lon_resolution = 0.25
    geodesic=True
    e=0
    k_exp = 20
    thereshold=1


    Cluster_params = (
        f"geodesic_{geodesic}_"
        f"e{e:.2f}"
    )

    W_params = (
        f"geodesic_{geodesic}"
    )

    # Construct results directory path
    results_directory = f"{file_path}"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)




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


    dataset = nc.Dataset(geo_file_path, mode='r')

    #from m/s to m/day
    land_mask = dataset.variables['land_mask'][:]
    sit = dataset.variables['sit'][:,:,:]

    # Access coordinates
    latitude = dataset.variables['regrided_rot_lat'][:]  
    longitude = dataset.variables['regrided_rot_lon'][:]

    dataset.close()

    x=longitude
    y=latitude

    Fmap_path = file_path+'Fmap_geodesic_True_nclusters10_e0.00.npy'
    labels_path = file_path+'Clusters_labels_geodesic_True_nclusters10_e0.00.npy'
    # Load the Fmap array from the file
    Fmap = np.load(Fmap_path)  # ntime [lon,lat] ntrajectories
    IC = Fmap[0,:,:]  #Take the position of the trajectory IC
    # Load the labels of the clusters
    labels= np.load(labels_path)

    grid_labels = griddata((IC[0, :], IC[1, :]), labels, (x,y), method='nearest')
    fmap_mask = np.isnan(griddata((IC[0, :], IC[1, :]), labels, (x,y), method='linear'))
    borders_avg = borders_binary(grid_labels)
    borders_avg = np.where(fmap_mask,np.nan, borders_avg)
    borders_avg[borders_avg > 0] = 1

    k=1
    for n_clusters in range(11, 31):
        # Construct the file paths with the current value of t
        Fmap_path = file_path + f'Fmap_geodesic_True_nclusters{n_clusters}_e0.00.npy'
        labels_path = file_path + f'Clusters_labels_geodesic_True_nclusters{n_clusters}_e0.00.npy'
        
        # Load the Fmap array from the file
        Fmap = np.load(Fmap_path)  # ntime [lon,lat] ntrajectories
        IC = Fmap[0, :, :]  # Take the position of the trajectory IC
        
        # Load the labels of the clusters
        labels = np.load(labels_path)
        
        # Interpolate the labels onto the grid
        grid_labels = griddata((IC[0, :], IC[1, :]), labels, (x, y), method='nearest')
        fmap_mask = np.isnan(griddata((IC[0, :], IC[1, :]), labels, (x, y), method='linear'))
        
        # Calculate borders
        borders = borders_binary(grid_labels)
        borders = np.where(fmap_mask, np.nan, borders)
        #borders = np.ma.masked_where(borders_avg==0, borders)
        borders[borders > 0] = 1
        
        # Accumulate the borders
        borders_avg += borders
        k = k+1
    # After the loop, borders_avg will contain the accumulated borders

    borders_avg = np.where(borders_avg==0, np.nan, borders_avg)
    borders_avg = np.where(land_mask==1, np.nan, borders_avg)
    borders_avg = borders_avg/k
    
    ini_final_clusters_landmask_borders_seas(Fmap, borders_avg, results_directory+"clusters_borders_10to30_"+Cluster_params+".png", e, longitude, latitude, land_mask,land_mask)