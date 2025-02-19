#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=special_case_W
#SBATCH --time=3-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output=wbeaufort_zoom_lead_febmars.out

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module list

# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate

parent_directory="/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"
year=2010
# Construct the input file path for the current year
input_files_directory="/cluster/projects/nn8008k/lluisa/NextSIM/OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice.nc"
  
echo "Processing file for year: $year"
# Run the Python script for each year
time srun python cartesian_to_rotated_latlon.py "$input_files_directory" \
"$parent_directory" > velice.txt
  
# Check the number of threads
env | grep NUM_THREADS

