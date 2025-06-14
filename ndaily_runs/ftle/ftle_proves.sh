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


directory="/cluster/projects/nn8008k/lluisa/NextSIM/seas/"
# Define variables
Ncores=32
ndays=10
# Time step-size (in days)
dt=0.1 # float
# Spacing of meshgrid (in degrees)
dx=0.1 # float
dy=0.1 # float
# Define ratio of auxiliary grid spacing vs original grid_spacing
aux_grid_ratio=.01 # float between [1/100, 1/5]
tmin=0

# Loop over years and seasons
year=$1
season=$2

#year=2009
#season="AMJ"

for tmin in $(seq 0 4 $((360 - ndays * 4))); do
    echo "$tmin"
    tmax=$((tmin + ndays * 4))
    echo "$tmax"

    formatted_dx=$(printf "%.3f" "$dx")
    formatted_dy=$(printf "%.3f" "$dy")
    formatted_dt=$(printf "%.3f" "$dt")
    formatted_aux_grid_ratio=$(printf "%.3f" "$aux_grid_ratio")

    FTLE_params="cleaned_FTLE_"
    FTLE_params+="${year}_${season}_"
    FTLE_params+="dx${formatted_dx}_"
    FTLE_params+="dy${formatted_dy}_"
    FTLE_params+="dt${formatted_dt}_"
    FTLE_params+="grid_ratio${formatted_aux_grid_ratio}"

    # Loop over years and seasons
    results_directory="${directory}/Fmap_${ndays}days/${FTLE_params}" 
 
    # Calculate the file number from tmin
    file_number=$((tmin / 4))
    Fmap_file_check="${results_directory}/${file_number}_FTLE.npy"

    # Check if the file exists
    if [[ -f "$Fmap_file_check" ]]; then
        echo "File $Fmap_file_check allready  exist. Skipping..."
    else
        echo "Generating $Fmap_file_check"
        time srun python /cluster/home/llpui9007/Programs/FTLE/FTLEIce_cleaned.py \
            "$Ncores" \
            "$ndays"\
            "$dt" \
            "$dx" \
            "$dy" \
            "$aux_grid_ratio" \
            "$results_directory"\
            "$tmin" \
            "$year" \
            "$season" > output/Fmap_${year}_${season}.txt
    fi
done
env | grep NUM



