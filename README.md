# Spectral clustering for Sea Ice

This Python script analyzes and visualizes ice flow patterns using scientific datasets. It supports tasks such as grid interpolation, trajectory computation, and clustering analysis.

## Features

- Handles large datasets in NetCDF format.
- Computes trajectories and interpolates irregular grids.
- Supports spectral clustering for data analysis.
- Visualizes results with custom plots and animations.

## Prerequisites

Before running the script, ensure you have the following installed: Python 3.8 or higher, and optionally a virtual environment.

## Installation

Clone this repository: `git clone <repository-url>`

Navigate to the project directory: `cd ice-grid-analysis`

Create and activate a virtual environment:
    For Linux/macOS: `python3 -m venv venv` followed by `source venv/bin/activate`
    For Windows: `python -m venv venv` followed by `venv\Scripts\activate`

Install dependencies: `pip install -r requirements.txt`

Run the script: `python ice_grid.py`

Deactivate the virtual environment when done: `deactivate`

## Usage

Prepare the necessary datasets:

- `OPA-neXtSIM_CREG025_ILBOXE140_2010_ice.nc`
- `OPA-neXtSIM_CREG025_ILBOXE140_2010_gridU_oce.nc`
- `mesh_mask_CREG025_3.6_NoMed.nc`

These should be placed in the appropriate directories specified in the script.

Run the script: `python ice_grid.py`
Results will be saved in the specified results directory.

## File Structure

project/
├── ice_grid.py               # Main script
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── data/                     # Datasets (not included in the repository)
│   ├── OPA-neXtSIM_CREG025_ILBOXE140_2010_ice.nc
│   ├── OPA-neXtSIM_CREG025_ILBOXE140_2010_gridU_oce.nc
│   └── mesh_mask_CREG025_3.6_NoMed.nc
├── results/                  # Output results
└── utils/                    # Additional utility functions


## Notes

- The script is optimized for parallel processing; ensure sufficient computational resources.
- Configurations like dataset paths or parameters should be updated in the script as needed.
- We use seed 563 for demos

## Contributing

Feel free to open issues or submit pull requests to enhance the project.

## License
