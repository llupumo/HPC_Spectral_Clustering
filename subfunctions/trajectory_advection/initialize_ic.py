import numpy as np 

def initialize_ic(lat_min,lat_max,lon_min,lon_max,ic_resolution,vel_land_interpolator):
   # Generate the latitude and longitude values
   IC_lat = np.arange(lat_min, lat_max + ic_resolution, ic_resolution)
   IC_lon = np.arange(lon_min, lon_max + ic_resolution, ic_resolution)
   # Create a meshgrid
   IC_lon_grid, IC_lat_grid = np.meshgrid(IC_lon,IC_lat)
   # Print the shapes of the grids

   # vectorize initial conditions
   lon0 = IC_lon_grid.ravel() # array (Nx*Ny, )
   lat0 = IC_lat_grid.ravel() # array (Nx*Ny, )

   IC = np.array([lat0, lon0]) # array (2, Nx*Ny)
   #Initial conditions over the whole domain


   # Remove the points where the velocity arrays are 0. This means either land or null initial velocity and therefore we don't 
   # want to have IC there.
   mask_IC = vel_land_interpolator(np.transpose(IC))
   idx_mask_IC = np.where(mask_IC==0)[0]
   IC = IC[:,idx_mask_IC]
   
   return IC


"""
### Function to generate an array of random tuples
Nt = 10000

print("Defining random initial conditions and filtering masked and water values")
def generate_random_tuples(Nt, xmin, xmax, ymin, ymax, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set the seed for reproducibility
    x_vals = np.random.uniform(xmin, xmax, Nt)
    y_vals = np.random.uniform(ymin, ymax, Nt)
    tuples_array = np.array(list(zip(x_vals, y_vals)))
    return tuples_array

IC = generate_random_tuples(Nt, lat_min, lat_max, lon_min, lon_max,seed).transpose()
"""
