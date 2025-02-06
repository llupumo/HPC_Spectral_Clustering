# import sys/os
import sys, os

# get current directory
path = os.getcwd()

# get parent directory
parent_directory = os.path.sep.join(path.split(os.path.sep))

# add utils folder to current working path in order to access the functions
sys.path.append(parent_directory)

# Import numpy
import numpy as np

# function which computes particle velocity
from velocity import velocity

def integration_dFdt_regular_grid_old(time, x, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, verbose = False, linear = False):
    '''
    Wrapper for RK4_step(). Advances the flow field given by u, v velocities, starting from given initial conditions. 
    The initial conditions can be specified as an array. 
    
    Parameters:
        time:          array (Nt,),  time array  
        x:             array (2, Npoints),  array of ICs (#Npoints = Number of initial conditions)
        X:             array (NY, NX),  X-meshgrid (of complete data domain)
        Y:             array (NY, NX),  Y-meshgrid (of complete data domain)
        Interpolant_u: Interpolant object for u(x, t)
        Interpolant_v: Interpolant object for v(x, t)
        periodic:      list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate. Time is i=3.
        bool_unsteady: bool, specifies if velocity field is unsteady/steady
        time_data:     array(1,NT), time data
        verbose:       bool, if True, function reports progress at every 100th iteration
        linear:        bool, set to true if Interpolant_u and Interpolant_v are LNDI interpolant_unsteady_uneven_linear_...
    
    Returns:
        Fmap:          array (Nt, 2, Npoints), integrated trajectory (=flow map)
        dFdt:          array (Nt-1,2, Npoints), velocity along trajectories (=time derivative of flow map) 

    '''
    # reshape x
    x = x.reshape(2, -1)

    # Initialize arrays for flow map and derivative of flow map
    Fmap = np.zeros((len(time), 2, x.shape[1]))
    dFdt = np.zeros((len(time)-1, 2, x.shape[1]))
    
    # Step-size
    dt = time[1]-time[0] #constant timestep for Runge Kutta
    
    counter = 0

    # initial conditions
    Fmap[counter,:,:] = x
    
    # Runge Kutta 4th order integration with fixed step size dt
    for counter, t in enumerate(time[:-1]):
        if verbose:
            if counter%100000 == 0:
                print('Percentage completed: ', np.around((t-time[0])/(time[-1]-time[0])*100, 4))
        
        Fmap[counter+1,:, :], dFdt[counter,:,:] = RK4_step(t, Fmap[counter,:, :], dt, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, linear)
        
        # check if periodic in x
        #if periodic[0]:
        
        #    Fmap[counter+1,0,:] = (Fmap[counter+1, 0,:]-X[0,0])%(X[0, -1]-X[0, 0])+X[0,0]
    
        # check if periodic in y
        #if periodic[1]:
        
        #    Fmap[counter+1,1,:] = (Fmap[counter+1, 1, :]-Y[0,0])%(Y[-1, 0]-Y[0, 0])+Y[0,0]
    
        counter += 1
    
    
    return Fmap, dFdt

def integration_dFdt(time, x, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, verbose = False, linear = False, timemod=1):
    '''
    Wrapper for RK4_step(). Advances the flow field given by u, v velocities, starting from given initial conditions. 
    The initial conditions can be specified as an array. 
    
    Parameters:
        time:          array (Nt,),  time array  
        x:             array (2, Npoints),  array of ICs (#Npoints = Number of initial conditions)
        X:             array (NY, NX),  X-meshgrid (of complete data domain)
        Y:             array (NY, NX),  Y-meshgrid (of complete data domain)
        Interpolant_u: Interpolant object for u(x, t)
        Interpolant_v: Interpolant object for v(x, t)
        periodic:      list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate. Time is i=3.
        bool_unsteady: bool, specifies if velocity field is unsteady/steady
        time_data:     array(1,NT), time data
        verbose:       bool, if True, function reports progress at every 100th iteration
        timemod:       int, number of saved trajectories
        linear:        bool, set to true if Interpolant_u and Interpolant_v are LNDI interpolant_unsteady_uneven_linear_...
    
    Returns:
        Fmap:          array (Nt, 2, Npoints), integrated trajectory (=flow map)
        dFdt:          array (Nt-1,2, Npoints), velocity along trajectories (=time derivative of flow map) 
    '''
    # reshape x
    x = x.reshape(2, -1)

    # Initialize arrays for flow map and derivative of flow map
    if timemod == 1:
        Fmap = np.zeros((len(time) , 2, x.shape[1]))
        dFdt = np.zeros((len(time)-1, 2, x.shape[1]))

        # Step-size
        dt = time[1]-time[0] #constant timestep for Runge Kutta
        
        counter = 0

        # initial conditions
        Fmap[counter,:,:] = x
        
        # Runge Kutta 4th order integration with fixed step size dt
        for counter, t in enumerate(time[:-1]):
            if verbose:
                if counter%1000000 == 0:
                    print('Percentage completed: ', np.around((t-time[0])/(time[-1]-time[0])*100, 4))
            
            Fmap[counter+1,:, :], dFdt[counter,:,:] = RK4_step(t, Fmap[counter,:, :], dt, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, linear)
            
            # check if periodic in x
            #if periodic[0]:
            
            #    Fmap[counter+1,0,:] = (Fmap[counter+1, 0,:]-X[0,0])%(X[0, -1]-X[0, 0])+X[0,0]
        
            # check if periodic in y
            #if periodic[1]:
            
            #    Fmap[counter+1,1,:] = (Fmap[counter+1, 1, :]-Y[0,0])%(Y[-1, 0]-Y[0, 0])+Y[0,0]
        
            counter += 1

    else:
        Fmap = np.zeros(((len(time) // timemod)+1, 2, x.shape[1]))
        dFdt = np.zeros(((len(time) // timemod), 2, x.shape[1]))

        Fmap_step = np.zeros((2,x.shape[1]))
        dFdt_step = np.zeros((2,x.shape[1]))
        
        # Step-size
        dt = time[1]-time[0] #constant timestep for Runge Kutta
        
        counter_fil = 0

        # initial conditions
        Fmap[counter_fil,:,:] = x
        Fmap_step = x

        # Runge Kutta 4th order integration with fixed step size dt
        for counter, t in enumerate(time[:-1]):
            if verbose:
                if counter%50 == 0:
                    print('Percentage completed: ', np.around((t-time[0])/(time[-1]-time[0])*100, 4))
            
            Fmap_step, dFdt_step = RK4_step(t, Fmap_step, dt, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, linear)

            if ((counter+1)%timemod == 0) and (counter!=0):
                Fmap[counter_fil+1,:,:] = Fmap_step #advance one time step
                dFdt[counter_fil,:,:] = dFdt_step #velocity in which you advance
                counter_fil += 1
        
        
        # check if periodic in x
        #if periodic[0]:
        
        
        #    Fmap[counter+1,0,:] = (Fmap[counter+1, 0,:]-X[0,0])%(X[0, -1]-X[0, 0])+X[0,0]
    
        # check if periodic in y
        #if periodic[1]:
        
        #    Fmap[counter+1,1,:] = (Fmap[counter+1, 1, :]-Y[0,0])%(Y[-1, 0]-Y[0, 0])+Y[0,0]
        
    return Fmap, dFdt

def RK4_step(t, x1, dt, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, linear = False):
    '''
    Advances the flow field by a single step given by u, v velocities, starting from given initial conditions. 
    The initial conditions can be specified as an array. 
    
    Parameters:
        time:           array (Nt,),  time array  
        x1:             array (2, Npoints),  array of currents positions
        X:              array (NY, NX)  X-meshgrid
        Y:              array (NY, NX)  Y-meshgrid 
        Interpolant_u:  Interpolant object for u(x, t)
        Interpolant_v:  Interpolant object for v(x, t)
        periodic:       list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate. Time is i=3.
        bool_unsteady:  bool, specifies if velocity field is unsteady/steady
        dt              timestep of the advection (found as h in some literature)
        linear:         bool, set to true if Interpolant_u and Interpolant_v are LNDI interpolant_unsteady_uneven_linear_...
    
    Returns:

        y_update:       array (2, Npoints), updated position (=flow map) 
        y_prime_update: array (2, Npoints), updated velocity (=time derivative of flow map) 
    '''

    t0 = t
    
    # Compute x_prime at the beginning of the time-step by re-orienting and rescaling the vector field
    x_prime = velocity(t0, x1, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, linear)
    
    # compute derivative
    k1 = dt * x_prime

    # Update position at the first midpoint.
    x2 = x1 + .5 * k1
     
    # Update time
    t = t0+.5*dt
    
    # Compute x_prime at the first midpoint.
    x_prime = velocity(t, x2, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, linear)
    
    # compute derivative
    k2 = dt * x_prime

    # Update position at the second midpoint.
    x3 = x1 + .5 * k2
    
    # Update time
    t = t0+.5*dt
    
    # Compute x_prime at the second midpoint.
    x_prime = velocity(t, x3, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, linear)
    
    # compute derivative
    k3 = dt * x_prime
    
    # Update position at the endpoint.
    x4 = x1 + k3
    
    # Update time
    t = t0+dt
    
    # Compute derivative at the end of the time-step.
    x_prime = velocity(t, x4, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, linear) 
    
    # compute derivative
    k4 = dt * x_prime
    
    # Compute RK4 derivative
    y_prime_update = 1.0 / 6.0*(k1 + 2 * k2 + 2 * k3 + k4)
    
    # Integration y <-- y + y_prime*dt
    y_update = x1 + y_prime_update
    
    return y_update, y_prime_update/dt
