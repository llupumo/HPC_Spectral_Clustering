#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=W3daysAMJ2009
#SBATCH --time=3-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=W3days2009AMJ_false79.out
# Define variables for easy modification
Ncores=32
year="2009"
season="AMJ"
ndays=3

IC_resolution=0.5
dt=0.0025
DT=0.01
geodesic=False

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module list

# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate

formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
#######################################################################################################################
for tmin in $(seq 204 4 $((360 - ndays * 4-10))); do
  tmax=$((tmin + ndays * 4))
  echo "$tmax"
  Fmap_params="${year}_${season}_"
  Fmap_params+="ic${IC_resolution}_"
  Fmap_params+="dt${formatted_dt}_"
  Fmap_params+="DT${formatted_DT}"

  directory="/cluster/projects/nn8008k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
  input_files_directory="${directory}Fmap_${ndays}days/${Fmap_params}/"

  parent_directory="/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"

  results_directory=${input_files_directory}
  # Run the Python script
  time srun python W_ndays.py "$Ncores" \
    "$input_files_directory" \
    "$parent_directory" \
    "$results_directory" \
    "$geodesic"\
    "$tmin"  >> W3days2009AMJ_79.txt
done
env | grep NUM_THREADS










