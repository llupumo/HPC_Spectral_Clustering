#!/bin/bash
# Define the range of years and the periods
years=$(seq 2000 2010)
periods=("AMJ" "OND" "JFM" "JAS")
#years=2000
#periods="OND"
if [ ! -d "output" ]; then
    mkdir output
fi
# Loop over each year
for year in $years; do
  # Loop over each period
  for period in "${periods[@]}"; do
    # Call your command or script here
    echo "Processing year: $year, period: $period"
    
    # Example: Submit a job with sbatch
    sbatch --job-name="W_${year}_${period}" --output="output/W_martin_${year}_${period}.out" W_martin.sh $year $period
  done
done