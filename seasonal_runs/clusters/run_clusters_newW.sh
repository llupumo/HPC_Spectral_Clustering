#!/bin/bash
# Define the range of years and the periods
years=$(seq 2000 2008)
periods=("AMJ" "OND" "JFM" "JAS")
#years=2009
#periods="AMJ"
# Loop over each year
for year in $years; do
  # Loop over each period
  for period in "${periods[@]}"; do
    # Call your command or script here
    echo "Processing year: $year, period: $period"
    
    # Example: Submit a job with sbatch
    sbatch --job-name="newW_${year}_${period}" --output="./output/cl_newW_${year}_${period}.out" clusters_newW.sh $year $period
  done
done
