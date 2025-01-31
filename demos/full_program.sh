#!/bin/bash
# Define the parameters

current_directory=$(pwd)
# Get the parent directory
parent_directory=$(dirname "$current_directory")

velocities_file_path="${parent_directory}/Data/OPA-neXtSIM_CREG025_ILBOXE140_2010_ice_latlon_rot_jacob.nc"
results_directory="${parent_directory}/Data/"

tmin=0
tmax=24
lat_resolution=1
lon_resolution=1
dt=0.1
DT=2
freq=10
geodesic=True
e=0
n_clusters=0

#######################################################################################################################

Fmap_params="tmin${tmin}_"
Fmap_params+="tmax${tmax}_"
Fmap_params+="latlonres${lat_resolution}x${lon_resolution}_"
Fmap_params+="dt${dt}_"
Fmap_params+="DT${DT}"

Cluster_params="geodesic_${geodesic}_nclusters${n_clusters}_e${e}"


Fmap_file_path="${results_directory}Fmap_${Fmap_params}/"
W_file_path="${Fmap_file_path}"
Clusters_file_path="${W_file_path}/${Cluster_params}/"


#######################################################################################################################

Ncores=10
# Call the Python script or Jupyter notebook
python Fmap.py \
  "$Ncores" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$Fmap_file_path" \
  "$tmin" \
  "$tmax" \
  "$lat_resolution" \
  "$lon_resolution" \
  "$dt" \
  "$DT" \
  --freq "$freq"

########################################################################################################################

Ncores=10

# Call the Python script or Jupyter notebook
python W.py \
  "$Ncores" \
  "$Fmap_file_path" \
  "$parent_directory" \
  "$W_file_path" \
  "$tmin" \
  "$tmax" \
  "$lat_resolution" \
  "$lon_resolution" \
  "$dt" \
  "$DT" \
  "$geodesic" \
  --freq "$freq"



########################################################################################################################

Ncores=10
# Call the Python script or Jupyter notebook
python Clusters.py \
  "$Ncores" \
  "$Fmap_file_path" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$Clusters_file_path" \
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
