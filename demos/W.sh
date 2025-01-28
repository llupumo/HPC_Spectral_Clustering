#!/bin/bash
# Define the parameters

current_directory=$(pwd)
# Get the parent directory
parent_directory=$(dirname "$current_directory")

Ncores=10
tmin=0
tmax=360
lat_resolution=0.25
lon_resolution=0.25
dt=1
DT=10
freq=10
geodesic=False


Fmap_params="tmin${tmin}_"
Fmap_params+="tmax${tmax}_"
Fmap_params+="latlonres${lat_resolution}x${lon_resolution}_"
Fmap_params+="dt${dt}_"
Fmap_params+="DT${DT}"



file_path="${parent_directory}/Data/Fmap_${Fmap_params}/"
results_directory="${file_path}"

# Call the Python script or Jupyter notebook
python W.py \
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
  "$geodesic" \
  --freq "$freq"

