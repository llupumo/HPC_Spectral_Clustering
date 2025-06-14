#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=0F
#SBATCH --time=0-0:30:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=loop.out

Ncores=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module list

# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate



#######################################################################################################################


# Run the Python script
time srun python Clusters.py > "loop.txt"