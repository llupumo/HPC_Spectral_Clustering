import numpy as np
import time
import sys, os
from pyproj import Geod

# get current directory
path = os.getcwd()
# get parent directory
parent_directory = os.path.sep.join(path.split(os.path.sep)[:-2])
# add utils and subfunctions folders to current working path
sys.path.append(parent_directory+"/subfunctions/latlon_transform")
# Import function for the polar rotation
from polar_rotation import polar_rotation_rx 

def integral_vec(Fmap, dT, time_deltas, i, j, geodesic = False):
    if geodesic==False:
        # Compute pairwise distances at each time step
        distances = np.linalg.norm(Fmap[1:, :, i] - Fmap[1:, :, j], axis=1) + np.linalg.norm(Fmap[:-1, :, i] - Fmap[:-1, :, j], axis=1)
    else:
        # Define the WGS84 ellipsoid
        geod = Geod(ellps='WGS84')  #equivalent to +b=6356752 +a=6378137'

        #Go back to the non-rotated coordinates to compute the geodesic distances
        Fmap_lat,Fmap_lon = polar_rotation_rx(np.array(Fmap[:,1,:]),np.array(Fmap[:,0,:]),-90)  #Fmap[:,0,:] contains longitudes and Fmap[:,1,:] latitudes
        distances =  geod.inv(Fmap_lon[1:, i], Fmap_lat[1:, i], Fmap_lon[1:, j], Fmap_lat[1:, j])[2] + geod.inv(Fmap_lon[:-1, i], Fmap_lat[:-1, i], Fmap_lon[:-1, j], Fmap_lat[:-1, j])[2]  #distances in m
    

    
    # Multiply by time and sum
    result = np.sum((time_deltas / 2) * distances)/dT    
    return result


"""
def integral_vec(Fmap, timemap, i, j, geodesic = False):

    start_time = time.time()  # Start the timer
    dT = timemap[-1]-timemap[0]

    if geodesic==False:
        # Compute pairwise distances at each time step
        distances = np.linalg.norm(Fmap[1:, :, i] - Fmap[1:, :, j], axis=1) + np.linalg.norm(Fmap[:-1, :, i] - Fmap[:-1, :, j], axis=1)
    else:
        # Define the WGS84 ellipsoid
        geod = Geod(ellps='WGS84')  #equivalent to +b=6356752 +a=6378137'

        #Go back to the non-rotated coordinates to compute the geodesic distances
        Fmap_lat,Fmap_lon = polar_rotation_rx(np.array(Fmap[:,1,:]),np.array(Fmap[:,0,:]),-90)  #Fmap[:,0,:] contains longitudes and Fmap[:,1,:] latitudes
        distances =  geod.inv(Fmap_lon[1:, i], Fmap_lat[1:, i], Fmap_lon[1:, j], Fmap_lat[1:, j])[2] + geod.inv(Fmap_lon[:-1, i], Fmap_lat[:-1, i], Fmap_lon[:-1, j], Fmap_lat[:-1, j])[2]  #distances in m
    
    # Compute time differences
    time_deltas = timemap[1:] - timemap[:-1]
    
    # Multiply by time and sum
    result = np.sum((time_deltas / 2) * distances)/dT
    
    end_time = time.time()  # End the timer
    # Print the time taken
    print(f"Time to construct W: {end_time - start_time:.6f} seconds")
    
    return result
"""

