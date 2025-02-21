import sys, os
import numpy as np
#Import packages for geodesic distences
from pyproj import Geod
# Import package for parallel computing
from joblib import Parallel, delayed
# Import package for interpolation
from scipy.interpolate import griddata

##################################### new version directly computing the gradient in x, y directions ######################################
def borders_binary(grid_labels):
    """
    input: regried version of the labels at each point of a regular mesh
    output: regrided version of borders at each point of a regular mesh with 0 in the non-boundary regions and 1 in the boundary regions 
    """
    dy = np.abs(np.diff(grid_labels, axis=0))
    dx = np.abs(np.diff(grid_labels, axis=1))
    # Initialize borders matrix directly
    borders = np.zeros(grid_labels.shape)
    borders[:dy.shape[0], :] += dy
    borders[1:, :] += dy
    borders[:, :dx.shape[1]] += dx
    borders[:, 1:] += dx
    # Set all non-zero elements to 1
    borders[borders != 0] = 1
    return borders

##################################### old version with assessing distances between pairs of points ########################################

def gradient_matrix(IC,labels,i_batch,j_batch,geodesic=False,thereshold=1.5):
    w =  []
    for k in range(len(i_batch)):
        if (k%10000 == 0):
            print(k)
        s=gradient_labels(IC,labels,i_batch[k],j_batch[k],geodesic,thereshold)
        w = np.append(w,s)
    return w

def gradient_labels(IC, labels, i, j, geodesic , thereshold):
    if geodesic==False:
        # Compute pairwise distances at each time step
        distance = np.linalg.norm(IC[:, i] - IC[:, j])
    else:
        # Define the WGS84 ellipsoid
        geod = Geod(ellps='WGS84')  #equivalent to +b=6356752 +a=6378137'

        #Go back to the non-rotated coordinates to compute the geodesic distances
        IC[1,:], IC[0,:] = polar_rotation_rx(np.array(IC[1,:]),np.array(IC[0,:]),-90)  #IC[0,:] contains longitudes and IC[1,:] latitudes
        distance =  geod.inv(IC[0, i], IC[1, i], IC[0, j], IC[1, j])[2] #distances in m

    if distance<=thereshold:
        return np.abs(labels[i]-labels[j])
    else:
        return 0
    
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def calculating_borders(IC, labels, geodesic, thereshold, Ncores):
    print("Preparing the parallel loop to compute the Similarity matrix") 
    n = IC.shape[1]

    indices = np.tril_indices(n,0,n)

    I=indices[0]
    J=indices[1]

    I_batch = list(split(I, Ncores)) # list (Nx*Ny)
    J_batch = list(split(J, Ncores)) # list (Nx*Ny)
    print("Dimensions of W triangular")
    print(n*n/2+n/2)

    print("Length of the array with w values")
    print(I_batch[0].shape)

    print("Computing the similarity matrix with the parallel loop")
    results = Parallel(n_jobs=Ncores, verbose = 10)(delayed(gradient_matrix)(IC, labels, I_batch[i], J_batch[i], geodesic, thereshold) for i in range(len(I_batch)))
    gradients = results[0]
    for res in results[1:]:
        gradients = np.append(gradients, res)
    del(results)

    # Create an empty matrix of zeros with shape (n, n)
    gradients_mx = np.zeros((n, n))
    gradients_mx[indices] = gradients
    # Fill the upper triangular part 
    gradients_mx = gradients_mx + gradients_mx.T - np.diag(np.diag(gradients_mx))
    #np.fill_diagonal(gradients_mx, 0)
    borders = [1 if x!=0 else 0 for x in np.sum(gradients_mx,axis=1)]

    return borders