#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=2009OND
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
Ncores=32
year="2010"
season="OND"
## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module list

# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate

IC_resolution=0.5
dt=0.001
DT=0.1
geodesic=False

formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
#######################################################################################################################

Fmap_params="${year}_${season}_"
Fmap_params+="ic${IC_resolution}_"
Fmap_params+="dt${formatted_dt}_"
Fmap_params+="DT${formatted_DT}"

directory="/cluster/projects/nn8008k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
input_files_directory="${directory}Fmap_tests/${Fmap_params}/"

parent_directory="/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"

results_directory=${input_files_directory}

# Run the Python script
time srun python W.py "$Ncores" \
  "$input_files_directory" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" 
env | grep NUM_THREADS

wait 

geodesic=True
# Run the Python script
time srun python /cluster/work/users/llpui9007/monthly_runs/W.py "$Ncores" \
  "$input_files_directory" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" 
env | grep NUM_THREADS



