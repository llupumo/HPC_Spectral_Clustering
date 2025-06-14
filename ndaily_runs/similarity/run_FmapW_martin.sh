#!/bin/bash
# Define the range of years and the periods
#years=$(seq 2000 2009)

years=2009
period="JAS"
#periods=("OND" "JFM" "JAS")
# Loop over each year
for year in $years; do
  # Call your command or script here
  echo "Processing year: $year"
  
  # Example: Submit a job with sbatch
  sbatch --job-name="job_${year}" --output="output/W_${year}.out" FmapW_martin_fix.sh $year $period
done
