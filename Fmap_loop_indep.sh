#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=monthFmap_loop
#SBATCH --time=3-0:0:0
#SBATCH --output=month_Fmap_loop.out
# Define variables for easy modification
Ncores=32
geodesic=False
# Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module list
# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate
# Get the parent directory
parent_directory=/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering
IC_resolution=0.5
dt=0.0025
DT=0.1
formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
# Loop over years and seasons
for year in 2009 2010; do
  for season in AMJ OND JAS JFM; do
    filename="OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice_90Rx_${season}.nc"
    directory="/cluster/projects/nn9970k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
    velocities_file_path="${directory}${filename}"
    Fmap_params="${year}_${season}_"
    Fmap_params+="ic${IC_resolution}_"
    Fmap_params+="dt${formatted_dt}_"
    Fmap_params+="DT${formatted_DT}"
    Fmap_results_directory="${directory}Fmap/${Fmap_params}/"
    # Run the Fmap Python script on a new node for each iteration in parallel
    srun --nodes=1 --ntasks=1 --cpus-per-task=$Ncores time python Fmap.py \
      "$Ncores" \
      "$velocities_file_path" \
      "$parent_directory" \
      "$Fmap_results_directory" \
      "$IC_resolution" \
      "$dt" \
      "$DT" > month_Fmap_loop_${year}_${season}.txt &
  done
done
# Wait for all background jobs to finish
wait
env | grep NUM