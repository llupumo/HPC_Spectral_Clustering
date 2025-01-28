# Spectral clustering for Sea Ice

This Python script analyzes and visualizes ice flow patterns using sea ice velocity from the NextSIM model. 


## Prerequisites

Before running the script, ensure you have the following installed: Python 3.8 or higher. 
See requirements.txt file for necessary libraries 
- netCDF4==1.7.1.post2
- numpy==2.0.0
- matplotlib==3.9.2
- scikit-learn==1.5.2
- scipy==1.14.1
- joblib==1.4.2
- ipykernel==6.25.2
- ipynb==0.5.1
- geopy==2.4.1
- pyproj==3.7.0

## Installation

Clone this repository: `git clone <repository-url>`

Create and activate a virtual environment:
    `python3 -m venv venv` followed by `source venv/bin/activate`
    
Install dependencies: `pip install -r requirements.txt`

## Usage

Place the following dataset in the directory HPC_Spectral_Clustering/Data/
- OPA-neXtSIM_CREG025_ILBOXE140_2010_ice_latlon_rot_jacob.nc

Run the following executables in the following order:

- /home/llu/Programs/HPC_Spectral_Clustering/demos/Fmap.sh
- /home/llu/Programs/HPC_Spectral_Clustering/demos/W.sh
- /home/llu/Programs/HPC_Spectral_Clustering/demos/Clusters.sh

Results will be saved in the HPC_Spectral_Clustering/Data/ directory


## Notes

- The script is optimized for parallel processing
- Configurations like dataset paths or parameters should be updated in the script as needed. You can do that by modifying the executable scripts
    - /home/llu/Programs/HPC_Spectral_Clustering/demos/Fmap.sh
    - /home/llu/Programs/HPC_Spectral_Clustering/demos/W.sh
    - /home/llu/Programs/HPC_Spectral_Clustering/demos/Clusters.sh
- We use seed 563 for demos

## Contributing

Feel free to open issues or submit pull requests to enhance the project.

