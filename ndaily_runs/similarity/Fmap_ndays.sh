#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=Fmapdays
#SBATCH --time=3-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=2009OND.out
# Define variables for easy modification
Ncores=32
year="2009"
season="OND"
ndays=10
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
directory="/cluster/projects/nn9970k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
velocities_file_path="${directory}${filename}" 
#year=$(echo "$filename" | awk -F'_' '{print $4}')
#season=$(echo "$filename" | awk -F'_' '{print $7}' | sed 's/.nc$//')
IC_resolution=0.5
dt=0.0025
DT=0.01
formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
#######################################################################################################################
for tmin in $(seq 0 4 $((360 - ndays * 4))); do
  echo "$tmin"
  tmax=$((tmin + ndays * 4))
  echo "$tmax"
  Fmap_params="${year}_${season}_"
  Fmap_params+="ic${IC_resolution}_"
  Fmap_params+="dt${formatted_dt}_"
  Fmap_params+="DT${formatted_DT}"
  results_directory="${directory}Fmap_${ndays}days/${Fmap_params}/"
  # Run the Python script
  time srun python Fmap_ndays.py \
    "$Ncores" \
    "$velocities_file_path"\
    "$parent_directory" \
    "$results_directory" \
    "$tmin" \
    "$tmax" \
    "$IC_resolution" \
    "$dt" \
    "$DT"  > Fmap_${year}_${season}.txt
done
env | grep NUM_THREADS
