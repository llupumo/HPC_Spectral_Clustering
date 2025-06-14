import numpy as np
import sys, os
# Import package for parallel computing
from joblib import Parallel, delayed

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory+"/subfunctions/trajectory_advection")
sys.path.append(parent_directory+"/subfunctions/Parallelisation")

from parallelised_functions import split
from integration_dFdt import parallel_Fmap

def trajectory_advection(IC, time_data, Interpolant_u, Interpolant_v, lon_grid, lat_grid, timemod, dt, Ncores, melting=True):

    print("Advecting trajectories")

    # Initial time (in days)
    t0 = time_data[0,0]

    # Final time (in days)
    tN = time_data[0,-1]


    # NOTE: For computing the backward trajectories set: tN < t0 and dt < 0.
    time_adv = np.arange(t0, tN+dt, dt) # len(time) = N

    # Periodic boundary conditions
    periodic_x = False
    periodic_y = False
    periodic_t = False
    periodic = [periodic_x, periodic_y, periodic_t]

    # Unsteady velocity field
    bool_unsteady = True

    x0_batch = list(split(IC[1], Ncores)) # lon
    y0_batch = list(split(IC[0], Ncores)) # lat

    results = Parallel(n_jobs=Ncores, verbose = 0)(delayed(parallel_Fmap)(x0_batch[i], y0_batch[i], time_adv, lon_grid, lat_grid, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, timemod, verbose = False, linear=False, melting=True) for i in range(len(x0_batch)))
    #results is a two dimensional list. First dimension stands for NCores and the result of each core. Second dimension stands for [0] Fmap and [1] dFdt. Then we access an array of
    #ntime x latlon x ntrajectories

    time_adv_mod = time_adv[::timemod]

    Fmap = results[0][0]
    DFDt = results[0][1]

    for res in results[1:]:
        Fmap = np.append(Fmap, res[0], axis = 2)
        DFDt = np.append(DFDt, res[1], axis = 2)

    return Fmap, DFDt, time_adv_mod
