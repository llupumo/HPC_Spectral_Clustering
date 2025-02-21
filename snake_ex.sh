#!/usr/bin/env bash

#SBATCH --account=nn8008k
#SBATCH --job-name='snake'
#SBATCH --time=1-00:00:00
#SBATCH --nodes=8
#SBATCH --cpus-per-task=32
#SBATCH --output=snake.out

# safe bash settings
set -euf -o pipefail

# this makes snakefiles available
module load snakemake/8.4.2-foss-2023a


# make sure directory "results" exists
mkdir -p results


snakemake --jobs 8 --cores 32 --keep-going
