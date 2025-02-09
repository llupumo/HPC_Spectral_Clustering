import sys, os
import numpy as np

# get current directory
path = os.getcwd()
# get parent directory
parent_directory = os.path.sep.join(path.split(os.path.sep)[:-2])

# add utils and subfunctions folders to current working path
sys.path.append(parent_directory+"/subfunctions/latlon_transform")
sys.path.append(parent_directory+"/subfunctions/Similarity_matrix_construction")

# Import function for the polar rotation
from polar_rotation import polar_rotation_rx 
# Import function to compute pairwise distances between trajectories
from trajectory_distance import integral_vec

def similarity_matrix(Fmap,i_batch,j_batch,K,timemap,geodesic):
    w =  []
    dT = timemap[-1]-timemap[0]
    # Compute time differences
    time_deltas = timemap[1:] - timemap[:-1]
    for k in range(len(i_batch)):
        if (k%int(len(i_batch)/2) == 0):
            print(k)
        if i_batch[k] == j_batch[k] :  #diagonal elements (same trajectory)
            w = np.append(w,K)
        else:
            s=integral_vec(Fmap,dT, time_deltas,i_batch[k],j_batch[k],geodesic)
            if s==0:                     #null disatnce between trajectories
                w = np.append(w,K)
            else:
                w = np.append(w,(1/s))

    #print("number of s = 0 is "+ str(m))
    #print("number of diagonals is "+ str(n))
        
    return w
