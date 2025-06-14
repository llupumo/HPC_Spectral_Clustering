#!/bin/bash
# Define the range of years and the periods
years=$(seq 2000 2010)
periods=("AMJ" "OND" "JFM" "JAS")
# Loop over each year
if [ ! -d "output" ]; then
    mkdir output
fi
for year in $years; do
  # Loop over each period
  for period in "${periods[@]}"; do
    # Call your command or script here
    echo "Processing year: $year, period: $period"
    
    # Example: Submit a job with sbatch
    sbatch --job-name="job_${year}_${period}" --output="output/Fmap_loop_${year}_${period}.out" Fmap_loop.sh $year $period
  done
done