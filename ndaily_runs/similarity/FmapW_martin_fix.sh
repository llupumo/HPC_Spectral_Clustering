#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=W10T10OND
#SBATCH --time=0-16:00:0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=32 #don't reverse ntasks per node with cpus per task!! it would not work
#SBATCH --cpus-per-task=1 
#SBATCH --output=w10t2010OND.out
# Define variables for easy modification

Ncores_Fmap=32
ndays=10
# Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module list
# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate
module load Python/3.12.3-GCCcore-13.3.0
module load mpi4py/3.1.5-gompi-2023b      

# Get the parent directory
parent_directory=/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering
IC_resolution=0.5
dt=0.0025
DT=0.01
formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")

# Loop over years and seasons
year=$1
season=$2


directory="/cluster/projects/nn8008k/lluisa/NextSIM/seas/"
filename="OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice_90Rx_${season}.nc"
velocities_file_path="${directory}${filename}"
#for tmin in $(seq 0 4 $((360 - ndays * 4))); do
tmin=56
echo "$tmin"
tmax=$((tmin + ndays * 4))
echo "$tmax"
Fmap_params="${year}_${season}_"
Fmap_params+="ic${IC_resolution}_"  
Fmap_params+="dt${formatted_dt}_"
Fmap_params+="DT${formatted_DT}"
Fmap_results_directory="${directory}Fmap_${ndays}days/${Fmap_params}/"

# Calculate the file number from tmin
file_number=$((tmin / 4))

W_results_directory=${Fmap_results_directory}
time srun python W_MPI.py "$Fmap_results_directory" \
"$parent_directory" \
"$W_results_directory" \
"$tmin" > output/W_${year}_${season}.txt

env | grep NUM
