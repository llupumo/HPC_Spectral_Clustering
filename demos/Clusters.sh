#!/bin/bash
# Define the parameters
current_directory=$(pwd)
# Get the parent directory
parent_directory=$(dirname "$current_directory")

Ncores=10
geo_file_path="${parent_directory}/Data/OPA-neXtSIM_CREG025_ILBOXE140_2010_ice_latlon_rot_jacob.nc"
file_path=${parent_directory}

tmin=0
tmax=360
lat_resolution=0.25
lon_resolution=0.25
dt=1
DT=10
freq=10
geodesic=False
e=0
n_clusters=0

Fmap_params="tmin${tmin}_"
Fmap_params+="tmax${tmax}_"
Fmap_params+="latlonres${lat_resolution}x${lon_resolution}_"
Fmap_params+="dt${dt}_"
Fmap_params+="DT${DT}"

Cluster_params="geodesic_${geodesic}_nclusters${n_clusters}_e${e}"


file_path="${parent_directory}/Data/Fmap_${Fmap_params}/"
results_directory="${file_path}/${Cluster_params}/"

# Call the Python script or Jupyter notebook
python Clusters.py \
  "$Ncores" \
  "$file_path" \
  "$geo_file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$tmin" \
  "$tmax" \
  "$lat_resolution" \
  "$lon_resolution" \
  "$dt" \
  "$DT" \
  "$geodesic" \
  --freq "$freq" \
  --e "$e"\
  --n_clusters "$n_clusters"

