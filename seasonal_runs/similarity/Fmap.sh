#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=Fmapmonths
#SBATCH --time=0-0:30:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=2010_AMJ.out
Ncores=32
year="2010"
season="AMJ"
IC_resolution=0.5
dt=0.0025
DT=0.1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module list

# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate


# Get the parent directory
parent_directory=/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering

formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
# Loop over years and seasons

filename="OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice_90Rx_${season}.nc"
directory="/cluster/projects/nn9970k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
velocities_file_path="${directory}${filename}"
Fmap_params="${year}_${season}_"
Fmap_params+="ic${IC_resolution}_"
Fmap_params+="dt${formatted_dt}_"
Fmap_params+="DT${formatted_DT}"
Fmap_results_directory="${directory}Fmap/${Fmap_params}/"

# Run the Python script
time srun python Fmap.py \
  "$Ncores" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$Fmap_results_directory" \
  "$IC_resolution" \
  "$dt" \
  "$DT" > Fmap_${year}_${season}.txt

env | grep NUM_THREADS
