#!/bin/bash
# Define the parameters
#SBATCH --account=nn8008k
#SBATCH --job-name=clu_Avg0
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=cluster_avg0_OND.out




# Activate a virtual environment if needed
source /cluster/home/llpui9007/venvs/Spectral_clustering_venv/bin/activate


python Borders_diff.py  > cluster_avg_short.txt
