#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name= Sim_mat
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module list

# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate



# Define the parameters
current_directory=$(pwd)
# Get the parent directory
parent_directory=$(dirname "$current_directory")

Ncores=10
geo_file_path="${parent_directory}/Data/OPA-neXtSIM_CREG025_ILBOXE140_2010_ice_latlon_rot_jacob.nc"
file_path=${parent_directory}

tmin=0
tmax=None
IC_resolution=0.5
dt=0.001
DT=0.1
freq=10
geodesic=True
e=0
n_clusters=0

formatted_e=$(printf "%.2f" "$e")
formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
#######################################################################################################################

Fmap_params="IC_res${IC_resolution}_"
Fmap_params+="dt${formatted_dt}_"
Fmap_params+="DT${formatted_DT}"

Cluster_params="geodesic_${geodesic}_nclusters${n_clusters}_e${e}"

formatted_e=$(printf "%.2f" "$e")
Cluster_params="geodesic_${geodesic}_nclusters${n_clusters}_e${formatted_e}"


file_path="${parent_directory}/Data/Fmap_${Fmap_params}/"
results_directory="${file_path}/${Cluster_params}/"

# Run the Python script
time srun python W.py \
  "$Ncores" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$Fmap_file_path" \
  "$tmin" \
  "$tmax" \
  "$IC_resolution" \
  "$dt" \
  "$DT" \
  "$geodesic" \
  --freq "$freq" \
  --e "$e"\
  --n_clusters "$n_clusters"
env | grep NUM_THREADS
