import numpy as np

def outflow_detector(IC,latmin,latmax,lonmin,lonmax):
    mask = (IC[0, :] >= latmin) & (IC[0, :] <= latmax) & (IC[1, :] >= lonmin) & (IC[1, :] <= lonmax)
    print("The number of trajectories outside the domain is: "+str(np.sum(mask==True)))
    return IC[:, ~mask]