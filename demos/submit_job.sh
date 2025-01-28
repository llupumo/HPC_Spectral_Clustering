#!/bin/bash
#SBATCH --account=t
#SBATCH --job-name=IceXYGrid
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10

# Activate a virtual environment if needed
source /home/llu/venvs/Spectral_clustering_venv/bin/activate
# Run the Python script
python ice_xy_grid.py
