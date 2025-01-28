#!/bin/bash
# Define the parameters

current_directory=$(pwd)
# Get the parent directory
parent_directory=$(dirname "$current_directory")

Ncores=10
file_path="${parent_directory}/Data/OPA-neXtSIM_CREG025_ILBOXE140_2010_ice_latlon_rot_jacob.nc"
results_directory="${parent_directory}/Data/"

tmin=0
tmax=360
lat_resolution=0.25
lon_resolution=0.25
dt=1
DT=10
freq=10

Fmap_params="tmin${tmin}_"
Fmap_params+="tmax${tmax}_"
Fmap_params+="latlonres${lat_resolution}x${lon_resolution}_"
Fmap_params+="dt${dt}_"
Fmap_params+="DT${DT}"

results_directory+="Fmap_${Fmap_params}/"

# Call the Python script or Jupyter notebook
python Fmap.py \
  "$Ncores" \
  "$file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$tmin" \
  "$tmax" \
  "$lat_resolution" \
  "$lon_resolution" \
  "$dt" \
  "$DT" \
  --freq "$freq"
