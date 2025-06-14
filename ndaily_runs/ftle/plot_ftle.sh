#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --time=3-00:20:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 #16
# Define variables for easy modification

# Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module list
# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load Cartopy/0.20.3-foss-2022a

# Loop over years and seasons
year=$1
season=$2


time srun plot_ftle.py \
    "$year" \
    "$season" > output/plotflte_${year}_${season}.txt

env | grep NUM