#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=recreate_test #monthFmapW
#SBATCH --time=0-0:20:0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=32 #2
#SBATCH --cpus-per-task=1 #16
#SBATCH --output=geod_True.out #FmapW.out
# Define variables for easy modification

# Define variables for easy modification
NcoresFmap=32
NcoresW=1
geodesic=True #False #True

# Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module list
# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate
module load Python/3.12.3-GCCcore-13.3.0
module load mpi4py/3.1.5-gompi-2023b            
#pip install -r /cluster/home/llpui9007/Programs/HPC_Spectral_Clustering/requirements.txt

# Get the parent directory
parent_directory=/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering
IC_resolution=0.5
dt=0.0025
DT=0.1
formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
# Loop over years and seasons
for year in 2009 2010; do #2010; do
  for season in AMJ JAS OND JFM; do # JAS OND JFM; do
    filename="OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice_90Rx_${season}.nc"
    directory="/cluster/projects/nn9970k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
    velocities_file_path="${directory}${filename}"
    Fmap_params="${year}_${season}_"
    Fmap_params+="ic${IC_resolution}_"
    Fmap_params+="dt${formatted_dt}_"
    Fmap_params+="DT${formatted_DT}"
    Fmap_results_directory="${directory}Fmap/${Fmap_params}/"
    # Run the Fmap Python script
    #time srun --nodes 1 --ntasks 1 --cpus-per-task 32 python Fmap.py \
    #  "$NcoresFmap" \
    #  "$velocities_file_path" \
    #  "$parent_directory" \
    #  "$Fmap_results_directory" \
    #  "$IC_resolution" \
    #  "$dt" \
    #  "$DT" > test_no_par.txt #FmapW_${year}_${season}.txt
    W_results_directory=${Fmap_results_directory}
    # Run the W Python script
    time srun python W_MPI.py "$NcoresW" \
      "$Fmap_results_directory" \
      "$parent_directory" \
      "$W_results_directory" \
      "$geodesic"  > geod_True.txt #FmapW_${year}_${season}.txt
    done
  done
env | grep NUM
