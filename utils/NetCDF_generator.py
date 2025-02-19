import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np

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

def save_velocities_sicsit_to_netCDF(out_path,time_data,interpolated_siu,interpolated_siv, interpolated_sic, interpolated_sit,regrided_land_mask,lon_grid,lat_grid, coord_fillvalue, vel_fillvalue, sic_fillvalue, sit_fillvalue):
    # Create a new NetCDF file
    ncfile = Dataset(out_path, 'w', format='NETCDF4')
    y_size = lat_grid.shape[0]
    x_size = lat_grid.shape[1]
    time_size = time_data.shape[0]
    # Create dimensions
    ncfile.createDimension('time', None)
    ncfile.createDimension('y', y_size)
    ncfile.createDimension('x', x_size)
    # Create variables
    time_var = ncfile.createVariable('time', np.float64, ('time',))
    time_var.standard_name = "time"
    time_var.long_name = "simulation time"
    time_var.units = "days since 1900-01-01 00:00:00"
    time_var.calendar = "standard"
    time_var.bounds = "time_bnds"
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
