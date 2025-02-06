# import Rectangular bivariate spline from scipy
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import SmoothBivariateSpline as SBS
from scipy.interpolate import griddata

import numpy as np

def regrid_unsteady(lat_irr, lon_irr, U, V, lat_reg, lon_reg):
    print("Regriding for "+str(U.shape[2])+" time steps")
    # define u, v interpolants
    Interpolated_vel = [[], []]
    Interpolated_vel[0] = np.array([griddata((lat_irr.ravel(),lon_irr.ravel()), U[:, :, t].ravel(), (lat_reg.ravel(), lon_reg.ravel()), method='linear', rescale=False).reshape(lon_reg.shape) for t in range(U.shape[2])]).transpose(1,2,0)
    Interpolated_vel[1] = np.array([griddata((lat_irr.ravel(),lon_irr.ravel()), V[:, :, t].ravel(), (lat_reg.ravel(), lon_reg.ravel()), method='linear', rescale=False).reshape(lon_reg.shape) for t in range(V.shape[2])]).transpose(1,2,0)
    
    return Interpolated_vel

def interpolant_unsteady(X, Y, U, V, method = "cubic"):
    '''
    Unsteady wrapper for scipy.interpolate.RectBivariateSpline. Creates a list of interpolators for u and v velocities
    
    Parameters:
        X: array (Ny, Nx), X-meshgrid
        Y: array (Ny, Nx), Y-meshgrid
        U: array (Ny, Nx, Nt), U velocity
        V: array (Ny, Nx, Nt), V velocity
        method: Method for interpolation. Default is 'cubic', can be 'linear'
        
    Returns:
        Interpolant: list (2,), U and V  interpolators
    '''
    # Cubic interpolation
    if method == "cubic":
                
        kx = 3
        ky = 3
               
    # linear interpolation
    elif method == "linear":
            
        kx = 1
        ky = 1  
            
    # define u, v interpolants
    Interpolant = [[], []]
                    
    for j in range(U.shape[2]):
                
        Interpolant[0].append(RBS(Y[:,0], X[0,:], U[:,:,j], kx=kx, ky=ky))
        Interpolant[1].append(RBS(Y[:,0], X[0,:], V[:,:,j], kx=kx, ky=ky))
    
    return Interpolant

def interpolant_steady(X, Y, U, V, method = "cubic"):
    '''
    Steady wrapper for scipy.interpolate.RectBivariateSpline. Creates a list of interpolators for u and v velocities
    
    Parameters:
        X: array (Ny, Nx), X-meshgrid
        Y: array (Ny, Nx), Y-meshgrid
        U: array (Ny, Nx), U velocity
        V: array (Ny, Nx), V velocity
        method: Method for interpolation. Default is 'cubic', can be 'linear'
        
    Returns:
        Interpolant: list (2,), U and V  interpolators
    '''
    # Cubic interpolation
    if method == "cubic":
                
        kx = 3
        ky = 3
               
    # linear interpolation
    elif method == "linear":
            
        kx = 1
        ky = 1
            
    # define u, v interpolants
    Interpolant = []
                
    Interpolant.append(RBS(Y[:,0], X[0,:], U, kx=kx, ky=ky))
    Interpolant.append(RBS(Y[:,0], X[0,:], V, kx=kx, ky=ky))  
        
    return Interpolant

from scipy.interpolate import LinearNDInterpolator as LNDI
def interpolant_unsteady_uneven_linear(X, Y, U, V):
            
    # define u, v interpolants
    Interpolant = [[], []]

    for i in range(U.shape[2]):   
        print(i)       
        Interpolant[0].append(LNDI(list(zip(X.ravel(), Y.ravel())), U[:,:,i].ravel(),fill_value=0))
        Interpolant[1].append(LNDI(list(zip(X.ravel(), Y.ravel())), V[:,:,i].ravel(),fill_value=0))
    
    return Interpolant

from scipy.interpolate import LinearNDInterpolator as LNDI
def regrid_unsteady_uneven_linear(X, Y, U, V,X_reg, Y_reg):
   
            
    # define u, v interpolants
    Interpolant = [[], []]

    for i in range(U.shape[2]):   
        print(i)       
        Interpolant[0].append(LNDI(list(zip(X.ravel(), Y.ravel())), U[:,:,i].ravel(),fill_value=0)(X.ravel(),Y.ravel()))
        Interpolant[1].append(LNDI(list(zip(X.ravel(), Y.ravel())), V[:,:,i].ravel(),fill_value=0)(X.ravel(),Y.ravel()))
    
    return Interpolant

def interpolant_unsteady_uneven_linear_masked(X, Y, U, V):
 
    # define u, v interpolants
    Interpolant = [[], []]
    print(Y.ravel().shape)
    print(X.ravel().shape)

    for i in range(U.shape[1]):   
        print(U[:,i].ravel().shape)       
        Interpolant[0].append(LNDI(list(zip(X.ravel(), Y.ravel())), U[:,i].ravel()))
        Interpolant[1].append(LNDI(list(zip(X.ravel(), Y.ravel())), V[:,i].ravel()))
    
    return Interpolant


from scipy.interpolate import SmoothBivariateSpline as SBS
def interpolant_unsteady_uneven(X, Y, U, V, method = "cubic"):
  
    # Cubic interpolation
    if method == "cubic":
                
        kx = 3
        ky = 3
               
    # linear interpolation
    elif method == "linear":
            
        kx = 1
        ky = 1  
            
    # define u, v interpolants
    Interpolant = [[], []]
    print(Y.ravel().shape)
    print(X.ravel().shape)

    for i in range(U.shape[2]):   
        print(U[:,:,i].ravel().shape)       
        Interpolant[0].append(SBS(Y.ravel(), X.ravel(), U[:,:,i].ravel(), kx=kx, ky=ky))
        Interpolant[1].append(SBS(Y.ravel(), X.ravel(), V[:,:,i].ravel(), kx=kx, ky=ky))
    
    return Interpolant