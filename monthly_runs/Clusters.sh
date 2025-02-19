#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=cluTest
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_clusters_AMJ2010.out

Ncores=1
year="2010"
season="AMJ"

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module list

# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate


IC_resolution=0.5
dt=0.0025
DT=0.1
freq=1
geodesic=False
e=0
n_clusters=0

formatted_e=$(printf "%.2f" "$e")
formatted_DT=$(printf "%.4f" "$DT")
formatted_dt=$(printf "%.4f" "$dt")
#######################################################################################################################


Fmap_params="${year}_${season}_"
Fmap_params+="ic${IC_resolution}_"
Fmap_params+="dt${formatted_dt}_"
Fmap_params+="DT${formatted_DT}"

Cluster_params="geodesic_${geodesic}_nclusters${n_clusters}_e${formatted_e}"

directory="/cluster/projects/nn8008k/lluisa/NextSIM/rotated_ice_velocities/seas/${season}/"
input_files_directory="${directory}Fmap/${Fmap_params}/"
parent_directory="/cluster/home/llpui9007/Programs/HPC_Spectral_Clustering"
results_directory=${input_files_directory}
filename="OPA-neXtSIM_CREG025_ILBOXE140_${year}_ice_90Rx_${season}.nc"
velocities_file_path="${directory}${filename}" 

#######################################################################################################################


# Run the Python script
time srun python Clusters.py \
  "$Ncores" \
  "$input_files_directory" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" \
  --e "$e"\
  --n_clusters "$n_clusters" > Clusters_${year}_${season}.txt


n_clusters=12
# Run the Python script
time srun python Clusters.py \
  "$Ncores" \
  "$input_files_directory" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" \
  --e "$e"\
  --n_clusters "$n_clusters" >> Clusters_${year}_${season}.txt

e=1.9
n_clusters=0
# Run the Python script
time srun python Clusters.py \
  "$Ncores" \
  "$input_files_directory" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" \
  --e "$e"\
  --n_clusters "$n_clusters" > Clusters_${year}_${season}.txt


n_clusters=12
# Run the Python script
time srun python Clusters.py \
  "$Ncores" \
  "$input_files_directory" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" \
  --e "$e"\
  --n_clusters "$n_clusters" >> Clusters_${year}_${season}.txt

e=1.5
n_clusters=0
# Run the Python script
time srun python Clusters.py \
  "$Ncores" \
  "$input_files_directory" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" \
  --e "$e"\
  --n_clusters "$n_clusters" > Clusters_${year}_${season}.txt


n_clusters=12
# Run the Python script
time srun python Clusters.py \
  "$Ncores" \
  "$input_files_directory" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" \
  --e "$e"\
  --n_clusters "$n_clusters" >> Clusters_${year}_${season}.txt

e=0.1
n_clusters=0
# Run the Python script
time srun python Clusters.py \
  "$Ncores" \
  "$input_files_directory" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" \
  --e "$e"\
  --n_clusters "$n_clusters" > Clusters_${year}_${season}.txt


n_clusters=12
# Run the Python script
time srun python Clusters.py \
  "$Ncores" \
  "$input_files_directory" \
  "$velocities_file_path" \
  "$parent_directory" \
  "$results_directory" \
  "$geodesic" \
  --e "$e"\
  --n_clusters "$n_clusters" >> Clusters_${year}_${season}.txt

env | grep NUM_THREADS

