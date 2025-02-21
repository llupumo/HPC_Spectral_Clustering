#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=clu_Avg0
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=cluster_avg0.out

Ncores=32
year="2009"
season="AMJ"
ndays=10
IC_resolution=0.5
dt=0.0025
DT=0.01
freq=1
geodesic=False
e=0
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




python Borders.py \
"$Ncores" \
"$input_files_directory" \
"$velocities_file_path" \
"$parent_directory" \
"$results_directory" \
"$geodesic" \
--e "$e" \
--n_clusters "$n_clusters" > cluster_avg_short.txt


