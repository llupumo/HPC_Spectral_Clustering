#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=W10F2009
#SBATCH --time=3-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=w10f2009_1task.out
# Define variables for easy modification
Ncores=32
ndays=10
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
DT=0.01
formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
for year in 2009; do
  for season in AMJ JAS OND JFM; do
    filename="OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice_90Rx_${season}.nc"
    directory="/cluster/projects/nn9970k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
    velocities_file_path="${directory}${filename}"
    for tmin in $(seq 0 4 $((360 - ndays * 4))); do
      tmax=$((tmin + ndays * 4))
      echo "$tmax"
      Fmap_params="${year}_${season}_"
      Fmap_params+="ic${IC_resolution}_"
      Fmap_params+="dt${formatted_dt}_"
      Fmap_params+="DT${formatted_DT}"
      Fmap_results_directory="${directory}Fmap_${ndays}days/${Fmap_params}/"
      W_results_directory="${Fmap_results_directory}"
      # Calculate the file number from tmin
      file_number=$((tmin / 4))
      file_check="${Fmap_results_directory}${file_number}_W_matrix_geodesic_False.png"
      # Check if the file exists
      if [[ -f "$file_check" ]]; then
        echo "File $file_check allready exists. Skipping..."
      else
        # Run the W Python script
        time srun python W_ndays.py "$Ncores" \
          "$Fmap_results_directory" \
          "$parent_directory" \
          "$W_results_directory" \
          "$geodesic" \
          "$tmin" >> 10days_FmapW.txt
      fi
    done
  done
done
env | grep NUM

# Loop over years and seasons

#for year in 2009; do
#  for season in AMJ JAS OND JFM; do
#    filename="OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice_90Rx_${season}.nc"
#    directory="/cluster/projects/nn9970k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
#    velocities_file_path="${directory}${filename}"
#    for tmin in $(seq 236 4 $((360 - ndays * 4))); do
#      tmax=$((tmin + ndays * 4))
#      echo "$tmin"
#      Fmap_params="${year}_${season}_"
#      Fmap_params+="ic${IC_resolution}_"
#      Fmap_params+="dt${formatted_dt}_"
#      Fmap_params+="DT${formatted_DT}"
#      Fmap_results_directory="${directory}Fmap_${ndays}days/${Fmap_params}/"
#     # Run the Fmap Python script
#     time srun python Fmap_ndays.py \
#        "$Ncores" \
#        "$velocities_file_path" \
#        "$parent_directory" \
#        "$Fmap_results_directory" \
#        "$tmin" \
#        "$tmax" \
#        "$IC_resolution" \
#        "$dt" \
#        "$DT" > FmapW_${year}_${season}_10days.txt
#      W_results_directory=${Fmap_results_directory}
#    done
#  done
#done
#wait






