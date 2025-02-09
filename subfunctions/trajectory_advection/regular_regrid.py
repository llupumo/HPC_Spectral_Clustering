import time
import numpy as np

#Import packages for interpolating
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator as LNDI


# Import package for parallel computing
from joblib import Parallel, delayed


def interpolate_frame(latitude, longitude, lat_grid, lon_grid,v, t):
    return griddata((latitude.ravel(), longitude.ravel()),v[:, :, t].ravel(),(lat_grid.ravel(), lon_grid.ravel()),method='linear',rescale=False).reshape(lon_grid.shape)

def regular_grid_interpolation(latitude, longitude, siu, siv,latitude_resolution,longitude_resolution, land_mask, Ncores):

    start_time = time.time()
    print("Interpolating to a regular grid")
    # Define the bounds of the grid
    lat_min, lat_max = latitude.min(), latitude.max()
    lon_min, lon_max = longitude.min(), longitude.max()

    # Generate the latitude and longitude values
    latitudes = np.arange(lat_min, lat_max + latitude_resolution, latitude_resolution)
    longitudes = np.arange(lon_min, lon_max + longitude_resolution, longitude_resolution)
    # Create a meshgrid
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Print the shapes of the grids
    print(f"Latitude grid shape: {lat_grid.shape}")
    print(f"Longitude grid shape: {lon_grid.shape}")

    #### Interpolate velocities and land_mask into the regular grid

    #Land mask: water 0, land 1
    mask_interpolator = LNDI(list(zip(latitude.ravel(), longitude.ravel())), land_mask.ravel(),fill_value=1)
    mask_interpol=mask_interpolator(lat_grid, lon_grid)


    # Parallelize the interpolation over the third dimension
    interpolated_data = Parallel(n_jobs=Ncores)(delayed(interpolate_frame)(latitude, longitude, lat_grid, lon_grid, siu, t) for t in range(siu.shape[2]))
    # Convert the list of arrays to a single numpy array and transpose
    interpolated_siu = np.array(interpolated_data).transpose(1, 2, 0)


    # Parallelize the interpolation over the third dimension
    interpolated_data = Parallel(n_jobs=Ncores)(delayed(interpolate_frame)(latitude, longitude, lat_grid, lon_grid, siv,t) for t in range(siv.shape[2]))
    # Convert the list of arrays to a single numpy array and transpose
    interpolated_siv = np.array(interpolated_data).transpose(1, 2, 0)


    interpolated_siu = np.ma.masked_array(interpolated_siu, mask=np.repeat(mask_interpol[:, :, np.newaxis], interpolated_siu.shape[2], axis=2))
    interpolated_siv = np.ma.masked_array(interpolated_siv, mask=np.repeat(mask_interpol[:, :, np.newaxis], interpolated_siu.shape[2], axis=2))

    #At this point the land values are masked but we want to have zeros instead of Nans to be able to advect


    # Set nan values to zero (in case there are any) so that we can apply interpolant. 
    # Interpolant does not work if the array contains nan values. 
    interpolated_siu[np.isnan(interpolated_siu)] = 0
    interpolated_siv[np.isnan(interpolated_siv)] = 0
    interpolated_siu = interpolated_siu.filled(0)
    interpolated_siv = interpolated_siv.filled(0)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Time taken for regriding the velocity field: {elapsed_time:.2f} seconds")

    return interpolated_siu, interpolated_siv, lat_grid, lon_grid, mask_interpol


