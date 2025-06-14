#!/bin/bash
# Define the range of years and the periods
#years=$(seq 2009 2010)
#periods=("AMJ" "OND" "JFM" "JAS")
years=2009 
periods="JAS"
# Loop over each year
for year in $years; do
  # Loop over each period
  for period in "${periods[@]}"; do
    # Call your command or script here
    echo "Processing year: $year, period: $period"
    # Example: Submit a job with sbatch
    sbatch --job-name="newW_${year}_${period}" --output="./output/newW_${year}_${period}.out" recalculate_similarity.sh $year $period
  done
done
