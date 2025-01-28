import netCDF4 as nc
from netCDF4 import Dataset
import sys, os
import time
import numpy as np
from numpy import ma
from numpy import pi

#Import packages for plotting
from matplotlib import pyplot as plt
from pylab import imshow,cm
#Import packages for interpolating
from scipy.interpolate import griddata

# Get the filename from command-line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <filename>")
    sys.exit(1)
input_data = sys.argv[1]
parent_directory = sys.argv[2]


#data directory
data_directory = os.path.sep.join(input_data.split(os.path.sep)[:-1])+"/"
#data file
file_name = os.path.sep.join(input_data.split(os.path.sep)[-1:]) 
# results directory
results_directory = data_directory+'rotated_ice_velocities/'

os.makedirs(results_directory, exist_ok=True)

# add utils and subfunctions folders to current working path
sys.path.append(parent_directory+"/utils")
sys.path.append(parent_directory+"/subfunctions/latlon_transform")

# Import function for the polar rotation
from ipynb.fs.defs.polar_rotation import polar_rotation_rx , cartesian_to_spherical, spherical_to_cartesian

# Import function to change units
from ipynb.fs.defs.convert_meters_per_second_to_deg_per_day import m_to_deg_r

# Open the NetCDF file
file_path = data_directory+file_name
dataset = nc.Dataset(file_path, mode='r')

metadata_path =  data_directory+'/mesh_mask_CREG025_3.6_NoMed.nc'
metadata = nc.Dataset(metadata_path, mode='r')

#Parameters to filter timesteps of the velocity vectors
tmin = 0
tmax = None
freq=1
#Parameters to filter area
ycut_min = 110
xcut_min = 0
xcut_max = 500

#Read sea ice velocities and directly transform from m/s to m/day
siu = dataset.variables['siu'][tmin:tmax,ycut_min:,xcut_min:xcut_max]*3600*24
siv = dataset.variables['siv'][tmin:tmax,ycut_min:,xcut_min:xcut_max]*3600*24
siu = np.transpose(siu, axes=(1, 2, 0))
siv = np.transpose(siv, axes=(1, 2, 0))
siu = siu[:,:,::freq]
siv = siv[:,:,::freq]

# Access specific variables
time_data = dataset.variables['time'][tmin:tmax]
time_data= np.reshape(time_data, (1,-1))
time_data = time_data[0,::freq]

# Access (lat,lon) coordinates in the t point
latitude_t = metadata.variables['nav_lat'][ycut_min:,xcut_min:xcut_max]
longitude_t = metadata.variables['nav_lon'][ycut_min:,xcut_min:xcut_max]

# Access (lat,lon) coordinates in the v point
longitude_v = metadata.variables['glamv'][0,ycut_min:,xcut_min:xcut_max]
latitude_v = metadata.variables['gphiv'][0,ycut_min:,xcut_min:xcut_max]

# Access (lat,lon) coordinates in the u point
longitude_u = metadata.variables['glamu'][0,ycut_min:,xcut_min:xcut_max]
latitude_u = metadata.variables['gphiu'][0,ycut_min:,xcut_min:xcut_max]

# Access grid spacing centered on the t-point from the metadata
e1t = metadata.variables['e1t'][0,ycut_min:,xcut_min:xcut_max]
e2t = metadata.variables['e2t'][0,ycut_min:,xcut_min:xcut_max]

# Close the datasets when done reading
dataset.close()
metadata.close()

#Rotation in the v-point
latitude_v_r, longitude_v_r = polar_rotation_rx(latitude_v,longitude_v,90)
#Rotation in the u-point
latitude_u_r, longitude_u_r = polar_rotation_rx(latitude_u,longitude_u,90)
#Rotation in the t-point
latitude_t_r, longitude_t_r = polar_rotation_rx(latitude_t,longitude_t,90)

# Discrete differenciation of the rotated coordinates in the u and v points along the x and y directions. 

dlonx_r = np.diff(longitude_u_r)[1:493,:]
dlony_r = np.diff(longitude_v_r,axis=0)[:,1:500]
dlatx_r = np.diff(latitude_u_r)[1:493,:]
dlaty_r = np.diff(latitude_v_r,axis=0)[:,1:500]

# Convert to arrays if they are matrices
dlatx_r = np.asarray(dlatx_r)
dlonx_r = np.asarray(dlonx_r)
dlaty_r = np.asarray(dlaty_r)
dlony_r = np.asarray(dlony_r)

# Mask the arrays
dlatx_r = np.ma.masked_array(dlatx_r, mask=np.ma.getmask(latitude_v[1:493,1:500]))
dlonx_r = np.ma.masked_array(dlonx_r, mask=np.ma.getmask(longitude_u[1:493,1:500]))
dlaty_r = np.ma.masked_array(dlaty_r, mask=np.ma.getmask(latitude_v[1:493,1:500]))
dlony_r = np.ma.masked_array(dlony_r, mask=np.ma.getmask(longitude_u[1:493,1:500]))

# Cut dimensions on the other variables
e1t = e1t[1:493,1:500]
e2t = e2t[1:493,1:500]
siu = siu[1:493,1:500]
siv = siv[1:493,1:500]

latitude_u_r = latitude_u_r[1:493,1:500]
longitude_u_r = longitude_u_r[1:493,1:500]
latitude_v_r = latitude_v_r[1:493,1:500]
longitude_v_r = longitude_v_r[1:493,1:500]
latitude_t_r = latitude_t_r[1:493,1:500]
longitude_t_r = longitude_t_r[1:493,1:500]


# Finish the computation of the forward differencing scheme. Note that computationalwise we do a forward differencing but in reality this corresponts to a central differenciation
# around the t-points
dlonxdx_r = dlonx_r/e1t
dlonydy_r = dlony_r/e2t
dlatxdx_r = dlatx_r/e1t
dlatydy_r = dlaty_r/e2t

############################################################## Interpolate values around the NP ######################################################################
# Define the row and column indices for positions to set to NaN (8x8 squared)
dlatxdx_r[300:310,280:290] = np.nan
dlatydy_r[300:310,280:290] = np.nan
dlonxdx_r[300:310,280:290] = np.nan
dlonydy_r[300:310,280:290] = np.nan

# Create a grid of coordinates
x, y = np.meshgrid(np.arange(dlatxdx_r.shape[1]), np.arange(dlatxdx_r.shape[0]))
# Masked array where the value is NaN
masked_array_dlatxdx = np.isnan(dlatxdx_r)
masked_array_dlatydy = np.isnan(dlatydy_r)
masked_array_dlonxdx = np.isnan(dlonxdx_r)
masked_array_dlonydy = np.isnan(dlonydy_r)

# Get the valid points (not NaN)
valid_points_dlatxdx = ~masked_array_dlatxdx
valid_points_dlatydy = ~masked_array_dlatydy
valid_points_dlonxdx = ~masked_array_dlonxdx
valid_points_dlonydy = ~masked_array_dlonydy

# Extract the coordinates and values of the valid points
valid_x_dlatxdx = x[valid_points_dlatxdx]
valid_y_dlatxdx = y[valid_points_dlatxdx]
valid_latx_values = dlatxdx_r[valid_points_dlatxdx]
valid_x_dlatydy = x[valid_points_dlatydy]
valid_y_dlatydy = y[valid_points_dlatydy]
valid_laty_values = dlatydy_r[valid_points_dlatydy]
valid_x_dlonxdx = x[valid_points_dlonxdx]
valid_y_dlonxdx = y[valid_points_dlonxdx]
valid_lonx_values = dlonxdx_r[valid_points_dlonxdx]
valid_x_dlonydy = x[valid_points_dlonydy]
valid_y_dlonydy = y[valid_points_dlonydy]
valid_lony_values = dlonydy_r[valid_points_dlonydy]

# Interpolate the missing value
dlatxdx_r_interpolated = griddata(
    (valid_x_dlatxdx, valid_y_dlatxdx), valid_latx_values, (x, y), method='cubic'
)
dlatydy_r_interpolated = griddata(
    (valid_x_dlatydy, valid_y_dlatydy), valid_laty_values, (x, y), method='cubic'
)
dlonxdx_r_interpolated = griddata(
    (valid_x_dlonxdx, valid_y_dlonxdx), valid_lonx_values, (x, y), method='cubic'
)
dlonydy_r_interpolated = griddata(
    (valid_x_dlonydy, valid_y_dlonydy), valid_lony_values, (x, y), method='cubic'
)

# Replace the original NaN value with the interpolated value
dlatxdx_r[masked_array_dlatxdx] = dlatxdx_r_interpolated[masked_array_dlatxdx]
dlatydy_r[masked_array_dlatydy] = dlatydy_r_interpolated[masked_array_dlatydy]
dlonxdx_r[masked_array_dlonxdx] = dlonxdx_r_interpolated[masked_array_dlonxdx]
dlonydy_r[masked_array_dlonydy] = dlonydy_r_interpolated[masked_array_dlonydy]

vlat_r = np.multiply(dlatxdx_r[:, :, np.newaxis],siu) + np.multiply(dlatydy_r[:, :, np.newaxis],siv)
vlon_r = np.multiply(dlonxdx_r[:, :, np.newaxis],siu) + np.multiply(dlonydy_r[:, :, np.newaxis],siv)


#Save the results in a NetCDF
out_path = results_directory+file_name[:-3]+'_90Rx.nc'
# Create a new NetCDF file
ncfile = Dataset(out_path, 'w', format='NETCDF4')
y_size = latitude_t_r.shape[0]
x_size = latitude_t_r.shape[1]
time_size = time_data.shape[0]
# Create dimensions
ncfile.createDimension('time', None)
ncfile.createDimension('y', y_size)
ncfile.createDimension('x', x_size)
#ncfile.createDimension('nv', 2)
# Create variables
time_var = ncfile.createVariable('time', np.float64, ('time',))
time_var.standard_name = "time"
time_var.long_name = "simulation time"
time_var.units = "days since 1900-01-01 00:00:00"
time_var.calendar = "standard"
time_var.bounds = "time_bnds"
#time_bnds_var = ncfile.createVariable('time_bnds', np.float64, ('time', 'nv'))
#time_bnds_var.units = "days since 1900-01-01 00:00:00"
latitude_var = ncfile.createVariable('rot_lat', np.float32, ('y','x'),fill_value=latitude_t.fill_value)
latitude_var.standard_name = " rotated latitude"
latitude_var.long_name = "latitude wrt the coordinate system rotated 90⁰ clockwise around the x axis. NP in y=-1, z=0"
longitude_var = ncfile.createVariable('rot_lon', np.float32, ('y','x'),fill_value=longitude_t.fill_value)
longitude_var.standard_name = " rotated longitude"
longitude_var.long_name = "longitude wrt the coordinate system rotated 90⁰ clockwise around the x axis. NP in y=-1, z=0"
vlon_var = ncfile.createVariable('vlon', np.float32, ('time','y', 'x'),fill_value=vlon_r.fill_value)
vlon_var.standard_name = "sea_ice_lon_velocity"
vlon_var.long_name = "Sea Ice Lon Velocity"
vlon_var.units = "deg day-1"
vlon_var.cell_methods = "time: mean (interval: 6 hours) area: mean"
vlat_var = ncfile.createVariable('vlat', np.float32, ('time', 'y', 'x'),fill_value=vlat_r.fill_value)
vlat_var.standard_name = "sea_ice_lat_velocity"
vlat_var.long_name = "Sea Ice Lat Velocity"
vlat_var.units = "deg day-1"
vlat_var.cell_methods = "time: mean (interval: 6 hours) area: mean"
# Assign data to variables
latitude_var[:,:] = latitude_t_r
longitude_var[:,:] = longitude_t_r
time_var[:] = time_data
vlat_var[:, :, :] = np.transpose(vlat_r, axes=(2,0,1))
vlon_var[:, :, :] = np.transpose(vlon_r, axes=(2,0,1))
# Add global attributes
ncfile.Conventions = "CF-1.6"
ncfile.institution = "UiT, Institute of Mathematics and Statistics, Tromsø"
ncfile.source = "neXtSIM model fields + cartesian_to_rotated_latlon.ipynb"
# Close the file
ncfile.close()
print("NetCDF file created successfully.")

