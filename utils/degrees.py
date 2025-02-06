import numpy as np

def deg_node(W,i):
    return sum(W[:,i])

def degree_matrix(W):
    return np.diag(deg_node(W,np.arange(W.shape[0])))

def degree_matrix_old(W):
    n=W.shape[1]
    D = np.zeros((n,n))
    for i in range(D.shape[0]):
        D[i,i]=deg_node(W,i)
    return D

