import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import sys, os

# get current directory
path = os.getcwd()
# get parent directory
parent_directory = os.path.sep.join(path.split(os.path.sep)[:-2])
sys.path.append(parent_directory+"/subfunctions/trajectory_advection")
from regular_regrid import regular_grid_interpolation, regular_grid_interpolation_scalar

def save_velocities_to_netCDF(out_path,time_data,interpolated_siu,interpolated_siv,regrided_land_mask,lon_grid,lat_grid,coord_fillvalue,vel_fillvalue):
    #Save the results in a NetCDF
    # Create a new NetCDF file
    ncfile = Dataset(out_path, 'w', format='NETCDF4')
    y_size = lat_grid.shape[0]
    x_size = lat_grid.shape[1]
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
    latitude_var = ncfile.createVariable('regrided_rot_lat', np.float32, ('y','x'),fill_value=coord_fillvalue)
    latitude_var.standard_name = "Regrided rotated latitude"
    latitude_var.long_name = "Regrided latitude wrt the coordinate system rotated 90⁰ clockwise around the x axis. NP in y=-1, z=0"
    longitude_var = ncfile.createVariable('regrided_rot_lon', np.float32, ('y','x'),fill_value=coord_fillvalue)
    longitude_var.standard_name = "Regrided rotated longitude"
    longitude_var.long_name = "Regrided longitude wrt the coordinate system rotated 90⁰ clockwise around the x axis. NP in y=-1, z=0"
    vlon_var = ncfile.createVariable('vlon', np.float32, ('time','y', 'x'),fill_value=vel_fillvalue)
    vlon_var.standard_name = "sea_ice_lon_velocity"
    vlon_var.long_name = "Sea Ice Lon Velocity"
    vlon_var.units = "deg day-1"
    vlon_var.cell_methods = "time: mean (interval: 6 hours) area: mean"
    vlat_var = ncfile.createVariable('vlat', np.float32, ('time', 'y', 'x'),fill_value=vel_fillvalue)
    vlat_var.standard_name = "sea_ice_lat_velocity"
    vlat_var.long_name = "Sea Ice Lat Velocity"
    vlat_var.units = "deg day-1"
    vlat_var.cell_methods = "time: mean (interval: 6 hours) area: mean"
    land_mask_var = ncfile.createVariable('land_mask', np.float32, ('y', 'x'),fill_value=vel_fillvalue)
    land_mask_var.standard_name = "land_mask"
    land_mask_var.long_name = "land_mask"
    land_mask_var.units = "binary"
    # Assign data to variables
    latitude_var[:,:] = lat_grid
    longitude_var[:,:] = lon_grid
    time_var[:] = time_data
    vlat_var[:, :, :] = np.transpose(interpolated_siv, axes=(2,0,1))
    vlon_var[:, :, :] = np.transpose(interpolated_siu, axes=(2,0,1))
    land_mask_var[:, :] = regrided_land_mask
    # Add global attributes
    ncfile.Conventions = "CF-1.6"
    ncfile.institution = "UiT, Institute of Mathematics and Statistics, Tromsø"
    ncfile.source = "neXtSIM model fields + cartesian_to_rotated_latlon.ipynb + regular_regrid.py"
    # Close the file
    ncfile.close()
    print("NetCDF file created successfully.")
    return 0

def save_velocities_sicsit_to_netCDF(out_path,time_data,interpolated_siu,interpolated_siv, interpolated_sic, interpolated_sit,regrided_land_mask,lon_grid,lat_grid, coord_fillvalue, vel_fillvalue, sic_fillvalue, sit_fillvalue, time_bnds_data):
    # Create a new NetCDF file
    ncfile = Dataset(out_path, 'w', format='NETCDF4')
    y_size = lat_grid.shape[0]
    x_size = lat_grid.shape[1]
    time_size = time_data.shape[0]
    # Create dimensions
    ncfile.createDimension('time', None)
    ncfile.createDimension('nv', 2)
    ncfile.createDimension('y', y_size)
    ncfile.createDimension('x', x_size)
    # Create variables
    time_var = ncfile.createVariable('time', np.float64, ('time',))
    time_var.standard_name = "time"
    time_var.long_name = "simulation time"
    time_var.units = "days since 1900-01-01 00:00:00"
    time_var.calendar = "standard"
    time_var.bounds = "time_bnds"
    time_bnds_var = ncfile.createVariable('time_bnds', np.float64, ('time','nv'))
    time_bnds_var.units = "days since 1900-01-01 00:00:00"
    latitude_var = ncfile.createVariable('regrided_rot_lat', np.float32, ('y', 'x'), fill_value=coord_fillvalue)
    latitude_var.standard_name = "Regrided rotated latitude"
    latitude_var.long_name = "Regrided latitude wrt the coordinate system rotated 90⁰ clockwise around the x axis. NP in y=-1, z=0"
    longitude_var = ncfile.createVariable('regrided_rot_lon', np.float32, ('y', 'x'), fill_value=coord_fillvalue)
    longitude_var.standard_name = "Regrided rotated longitude"
    longitude_var.long_name = "Regrided longitude wrt the coordinate system rotated 90⁰ clockwise around the x axis. NP in y=-1, z=0"
    vlon_var = ncfile.createVariable('vlon', np.float32, ('time', 'y', 'x'), fill_value=vel_fillvalue)
    vlon_var.standard_name = "sea_ice_lon_velocity"
    vlon_var.long_name = "Sea Ice Lon Velocity"
    vlon_var.units = "deg day-1"
    vlon_var.cell_methods = "time: mean (interval: 6 hours) area: mean"
    vlat_var = ncfile.createVariable('vlat', np.float32, ('time', 'y', 'x'), fill_value=vel_fillvalue)
    vlat_var.standard_name = "sea_ice_lat_velocity"
    vlat_var.long_name = "Sea Ice Lat Velocity"
    vlat_var.units = "deg day-1"
    vlat_var.cell_methods = "time: mean (interval: 6 hours) area: mean"
    land_mask_var = ncfile.createVariable('land_mask', np.float32, ('y', 'x'), fill_value=vel_fillvalue)
    land_mask_var.standard_name = "land_mask"
    land_mask_var.long_name = "land_mask"
    land_mask_var.units = "binary"
    # Create SIC and SIT variables
    sic_var = ncfile.createVariable('sic', np.float32, ('time', 'y', 'x'), fill_value=sic_fillvalue)
    sic_var.standard_name = "sea_ice_concentration"
    sic_var.long_name = "Sea Ice Concentration"
    sic_var.units = "1"  # Unitless, typically a fraction or percentage
    sic_var.cell_methods = "time: mean (interval: 6 hours) area: mean"
    sit_var = ncfile.createVariable('sit', np.float32, ('time', 'y', 'x'), fill_value=sit_fillvalue)
    sit_var.standard_name = "sea_ice_thickness"
    sit_var.long_name = "Sea Ice Thickness"
    sit_var.units = "m"
    sit_var.cell_methods = "time: mean (interval: 6 hours) area: mean"
    # Assign data to variables
    latitude_var[:, :] = lat_grid
    longitude_var[:, :] = lon_grid
    time_var[:] = time_data
    time_bnds_var[:,:] = time_bnds_data
    vlat_var[:, :, :] = np.transpose(interpolated_siv, axes=(2, 0, 1))
    vlon_var[:, :, :] = np.transpose(interpolated_siu, axes=(2, 0, 1))
    land_mask_var[:, :] = regrided_land_mask
    sic_var[:, :, :] = np.transpose(interpolated_sic, axes=(2, 0, 1))
    sit_var[:, :, :] = np.transpose(interpolated_sit, axes=(2, 0, 1))
    # Add global attributes
    ncfile.Conventions = "CF-1.6"
    ncfile.institution = "UiT, Institute of Mathematics and Statistics, Tromsø"
    ncfile.source = "neXtSIM model fields + cartesian_to_rotated_latlon.ipynb + regular_regrid.py"
    # Close the file
    ncfile.close()
    print("NetCDF file created successfully.")
    return 0

def generate_regrided(reg_vel_file_path,velocities_file_path,latitude_resolution,longitude_resolution,Ncores):
    # Read dataset
    print("Reading the input data")
    dataset = nc.Dataset(velocities_file_path, mode='r')
    #from m/s to m/day
    siu = dataset.variables['vlon'][:,:,:]
    siv = dataset.variables['vlat'][:,:,:]
    siu = np.transpose(siu, axes=(1, 2, 0))
    siv = np.transpose(siv, axes=(1, 2, 0))
    sic = dataset.variables['sic'][:,:,:]
    sit = dataset.variables['sit'][:,:,:]
    sic = np.transpose(sic, axes=(1, 2, 0))
    sit = np.transpose(sit, axes=(1, 2, 0))
    land_mask=siv[:,:,0].mask
    # Access coordinates
    latitude = dataset.variables['rot_lat'][:]  
    longitude = dataset.variables['rot_lon'][:]
    # Access specific variables
    time_data = dataset.variables['time'][:] 
    time_bnds_data = dataset.variables['time_bnds'][:,:]
    dataset.close()

    #### Define a regular grid both for the IC and to use to generate the interpolators
    interpolated_siu, interpolated_siv, lat_grid, lon_grid, regrided_land_mask = regular_grid_interpolation(latitude, longitude, siu, siv,latitude_resolution,longitude_resolution,land_mask,Ncores)
    interpolated_sic, lat_grid, lon_grid = regular_grid_interpolation_scalar(latitude, longitude, sic,latitude_resolution,longitude_resolution,land_mask,Ncores)
    interpolated_sit, lat_grid, lon_grid = regular_grid_interpolation_scalar(latitude, longitude, sit,latitude_resolution,longitude_resolution,land_mask,Ncores)
    vel_fillvalue = siu.fill_value
    coord_fillvalue = latitude.fill_value
    sic_fillvalue = -1.e+14
    sit_fillvalue = -1.e+14
    save_velocities_sicsit_to_netCDF(reg_vel_file_path,time_data,interpolated_siu,interpolated_siv, interpolated_sic, interpolated_sit, regrided_land_mask, lon_grid,lat_grid,coord_fillvalue,vel_fillvalue,sic_fillvalue,sit_fillvalue, time_bnds_data)
    
    print(f"Regrided velocity created.")
