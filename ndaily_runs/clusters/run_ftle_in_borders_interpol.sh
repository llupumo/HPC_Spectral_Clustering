#!/bin/bash
# Define the range of years and the periods
#years=$(seq 2009 2010)
#periods=("AMJ" "OND" "JFM" "JAS")
tmin=29
years=(2009)
periods=("AMJ")
# Loop over each year
for year in $years; do
  # Loop over each period
  for period in "${periods[@]}"; do
    # Call your command or script here
    echo "Processing year: $year, period: $period"
    
    # Example: Submit a job with sbatch
    sbatch --job-name="bf_${year}_${period}" --output="./output/borders_ftle_interpol${year}_${period}_${tmin}.out" ftle_in_borders_interpol.sh $year $period $tmin
  done
done
