import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# get current directory
path = os.getcwd()
# get parent directory
parent_directory = os.path.sep.join(path.split(os.path.sep)[:-2])
sys.path.append(parent_directory+"/utils")
from degrees import degree_matrix
from degrees import deg_node

def from_similarity_to_eigen(Fmap, W_vec,e,K,k_exp):
    n=Fmap.shape[2]
    indices = np.tril_indices(n,0,n)
    print("The percentage of spercified elements is "+str(np.sum(W_vec < e)/np.sum(W_vec)))
    W_vec[W_vec < e] = 0

    # Create an empty matrix of zeros with shape (n, n)
    W = np.zeros((n, n))
    W[indices] = W_vec
    # Fill the upper triangular part 
    W = W + W.T - np.diag(np.diag(W))

    D=degree_matrix(W)
    indices_to_remove = np.where(D == K)[0]
    print(indices_to_remove)
    print(str(indices_to_remove.shape)+" trajectories have been removed because they were not similar to any other trajectories")

    D = np.delete(np.delete(D, indices_to_remove, axis=0), indices_to_remove, axis=1)
    W = np.delete(np.delete(W, indices_to_remove, axis=0), indices_to_remove, axis=1)
    Fmap = np.delete(Fmap,indices_to_remove, axis=2)

    L=D-W
  
    #Note that D, W, and therefore L, are real symmetric matrices (required for function "scipy.linalg.eigh")

    print("Computing first "+str(k_exp)+" eigenvalues")
    l = eigh(L,D,eigvals_only=False,subset_by_index=[0,k_exp-1])
    l_vect = l[1]
    l = l[0]

    diff_l=list(np.diff(l))
    k=diff_l.index(max(diff_l))+1

    l_vect.shape

    #We start by cutting of the eigenspace for the number of clusters we want
    # set number of clusters; automatically to k

    print("k_means clustering")
    print("The default number of clusters is "+str(k))
    n_clusters=k

    return l_vect,l,Fmap,k

