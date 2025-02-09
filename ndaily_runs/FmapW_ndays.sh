#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=2009OND
#SBATCH --time=3-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --output=FmapW_2009_OND.out
# Define variables for easy modification

Ncores=32
year="2009"
season="OND"
ndays=3

geodesic=False

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module list
# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate
#current_directory=$(pwd)
# Get the parent directory
parent_directory=/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering #$(dirname "$current_directory")
filename="OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice_90Rx_${season}.nc"
directory="/cluster/projects/nn8008k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
velocities_file_path="${directory}${filename}" 
#year=$(echo "$filename" | awk -F'_' '{print $4}')
#season=$(echo "$filename" | awk -F'_' '{print $7}' | sed 's/.nc$//')
IC_resolution=0.5
dt=0.0025
DT=0.01
freq=1
formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
#######################################################################################################################
for tmin in $(seq 0 4 $((360 - ndays * 4))); do
  tmax=$((tmin + ndays * 4))
  echo "$tmax"
  Fmap_params="${year}_${season}_"
  Fmap_params+="ic${IC_resolution}_"
  Fmap_params+="dt${formatted_dt}_"
  Fmap_params+="DT${formatted_DT}"
  Fmap_results_directory="${directory}Fmap_${ndays}days/${Fmap_params}/"
  # Run the Python script
  time srun python Fmap_ndays.py \
    "$Ncores" \
    "$velocities_file_path"\
    "$parent_directory" \
    "$Fmap_results_directory" \
    "$tmin" \
    "$tmax" \
    "$IC_resolution" \
    "$dt" \
    "$DT" \
    --freq "$freq" > FmapW_${year}_${season}.txt

  W_results_directory=${Fmap_results_directory}
  # Run the Python script
  time srun python W_ndays.py "$Ncores" \
    "$Fmap_results_directory" \
    "$parent_directory" \
    "$W_results_directory" \
    "$geodesic"\
    "$tmin"  >> FmapW_${year}_${season}.txt
done
env | grep NUM_THREADS





