#!/bin/bash
# Define the range of years and the periods
#years=$(seq 2000 2010)
periods=("OND" "JFM" "JAS")
years=2009
#periods="AMJ"
# Loop over each year
for year in $years; do
  # Loop over each period
  for period in "${periods[@]}"; do
    # Call your command or script here
    echo "Processing year: $year, period: $period"
    
    # Example: Submit a job with sbatch
    sbatch --job-name="job_${year}_${period}" --output="output/Fmap_${year}_${period}.out" Fmap_martin.sh $year $period
  done
done
