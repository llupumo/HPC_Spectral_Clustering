#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --time=00-06:20:0
#SBATCH --nodes=1
#SBATCH --ntasks=1 #2
#SBATCH --cpus-per-task=32 #16
# Define variables for easy modification


# Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module list
# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate
#module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load Cartopy/0.20.3-foss-2022a


year=$1
season=$2
ndays=10


for tmin in $(seq 56 4 $((360 - ndays * 4))); do
    echo "Processing year: $year, period: $season, time: $tmin"
    time srun python recalculate_similarity.py  \
        "$year" \
        "$season" \
        "$tmin" > ./output/newW_${year}_${season}.txt
done
env | grep NUM

