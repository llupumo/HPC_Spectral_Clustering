#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=npnu
#SBATCH --time=3-00:20:0
#SBATCH --nodes=1
#SBATCH --ntasks=1 #2
#SBATCH --cpus-per-task=1 #16
#SBATCH --output=npnu.out
# Define variables for easy modification



# Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module list
# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate
module load Cartopy/0.20.3-foss-2022a

env | grep NUM

time srun python /cluster/work/users/llpui9007/ndaily_runs/ftle/numpynumba.py > npnu.txt