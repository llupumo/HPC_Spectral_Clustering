import numpy as np
from numpy import pi

def convert_meters_per_second_to_deg_per_day(X, Y, U_ms, V_ms):
    
    '''
    Converts units of velocity from m/s to deg/day. The units of the velocity field must 
    match the units of the grid coordinates and time.
    
    Parameters:
        X:       array(Ny, Nx), X-meshgrid.
        Y:       array(Ny, Nx), Y-meshgrid.
        U_ms:    array(Ny, Nx, Nt), x-component of velocity field in m/s
        V_ms:    array(Ny, Nx, Nt), y-component of velocity field in m/s
         
    Returns:
        U_degday:    array(Ny, Nx, Nt), x-component of velocity field in deg/day
        V_degday:    array(Ny, Nx, Nt), y-component of velocity field in deg/day
    '''
    
    # import numpy
    import numpy as np
    
    # import math tools
    from math import cos, pi
    
    # Velocity field
    U_degday, V_degday = np.nan*U_ms.copy(), np.nan*V_ms.copy()
    
    # Radius of the earth
    earthRadius = 6371*(10**3)
    
    # Iterate over grid
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            U_degday[i, j, :] = (U_ms[i, j, :] / (cos(Y[i,j]*(pi/180))*(earthRadius)))*180*3600*24/pi
            V_degday[i, j, :] = (V_ms[i, j, :] / earthRadius)*180*3600*24/pi

    return U_degday, V_degday



def m_to_deg_r(V_mday, U_mday, latitude):

    '''
    Converts units of velocity from m/day to deg/day. The units of the velocity field must 
    match the units of the grid coordinates and time. We account for the degrees of the rotated coordinate system. Therefore we account for the difference between the equator circumference and the polar circumference. The operations are optimized with 
    numpy array multiplication
    
    Parameters:
        latitude:  array(Ny, Nx), Y-meshgrid storing the latitudes
        V_mday:    array(Ny, Nx, Nt), lat-component of velocity field in m/day
        U_mday:    array(Ny, Nx, Nt), lon-component of velocity field in m/day
        
         
    Returns:
        V_degday:    array(Ny, Nx, Nt), lat-component of velocity field in deg/day
        U_degday:    array(Ny, Nx, Nt), lon-component of velocity field in deg/day
    '''

    eq_radius = 6378.1*(10**3) #m 
    eq_circ = 2*pi*eq_radius #m; circumference along the equator
    polar_radius = 6356.8*(10**3) #m
    polar_circ = 2*pi*polar_radius #m; circumference along the NP and SP

    V_degday = V_mday*360/eq_circ #polar_circ
    U_degday = np.multiply(np.cos(np.radians(latitude))[:,:,np.newaxis],U_mday)*360/eq_circ
    
    
    return V_degday, U_degday