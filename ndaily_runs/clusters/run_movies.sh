#!/bin/bash
# Define the range of years and the periods
#years=$(seq 2009 2010)
periods=("AMJ" "OND" "JFM" "JAS")
years=2009
#periods="OND"
ndays=10
tmins=$(seq 0 1 80)
#tmins=2
# Loop over each year
for year in $years; do
  # Loop over each period
  for period in "${periods[@]}"; do
	for tmin in $tmins; do
    		# Call your command or script here
    		echo "Processing year: $year, period: $period, tmin: $tmin"
    
    		# Example: Submit a job with sbatch
    		sbatch --job-name="${period}_${tmin}" --output="./output/movies_${year}_${period}_${tmin}.out" movies.sh $year $period $tmin
  
	done
    done
done
