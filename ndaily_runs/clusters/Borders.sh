#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=plot
#SBATCH --time=0-0:10:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=plot.out

# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate

# Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module list
# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate
module load Python/3.12.3-GCCcore-13.3.0   
module load Cartopy/0.20.3-foss-2022a

python Borders_diff.py  #> plot.txt


