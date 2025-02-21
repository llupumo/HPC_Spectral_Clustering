#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=clu3days
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=3days_clusters_AMJ2009e0.out

Ncores=1
year="2009"
season="AMJ"
ndays=3
IC_resolution=0.5
dt=0.0025
DT=0.01
freq=1
geodesic=False
e=0.1
n_clusters=12


# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate


formatted_e=$(printf "%.2f" "$e")
formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
#######################################################################################################################

Fmap_params="${year}_${season}_"
Fmap_params+="ic${IC_resolution}_"
Fmap_params+="dt${formatted_dt}_"
Fmap_params+="DT${formatted_DT}"

Cluster_params="geodesic_${geodesic}_nclusters${n_clusters}_e${formatted_e}"

directory="/cluster/projects/nn8008k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
parent_directory="/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"
filename="OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice_90Rx_${season}.nc"
Fmap_params="${year}_${season}_"
Fmap_params+="ic${IC_resolution}_"
Fmap_params+="dt${formatted_dt}_"
Fmap_params+="DT${formatted_DT}"
input_files_directory="${directory}Fmap_${ndays}days/${Fmap_params}/"
results_directory="${input_files_directory}geodesic_${geodesic}/"
velocities_file_path="${directory}${filename}" 
#######################################################################################################################¨


for tmin in $(seq 0 4 $((360 - ndays * 4))); do
  echo ${tmin}
  python Clusters_ndays_nopng.py \
    "$Ncores" \
    "$input_files_directory" \
    "$velocities_file_path" \
    "$parent_directory" \
    "$results_directory" \
    "$geodesic" \
    "$tmin" \
    --e "$e"\
    --n_clusters "$n_clusters" > 3clusters_AMJ2009.txt
  done



