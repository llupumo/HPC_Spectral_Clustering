# Import numpy
import numpy as np

def velocity(t, x, X, Y, Interpolant_u, Interpolant_v, periodic, bool_unsteady, time_data, linear = False):
    '''
    Evaluate the interpolated velocity field over the specified spatial locations at the specified time.
    
    Parameters:
        t:              float,  time instant  
        x:              array (2,Npoints),  array of ICs
        X:              array (NY, NX)  X-meshgrid of data domain
        Y:              array (NY, NX)  Y-meshgrid of data domain
        Interpolant_u:  Interpolant object for u(x, t)
        Interpolant_v:  Interpolant object for v(x, t)
        periodic:       list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate. Time is i=3.
        bool_unsteady:  bool, specifies if velocity field is unsteady/steady
        time_data:      array(1, NT) time of velocity data
        linear:         bool, set to true if Interpolant_u and Interpolant_v are LNDI interpolant_unsteady_uneven_linear_...
    Returns:

        vel:            array(2,Npoints), velocities, vel[0,:] --> x-coordinate of velocity, vel[1,:] --> y-coordinate of velocity
    '''
    x_eval = x.copy()
    
    # check if periodic in x
    if periodic[0]:
        
        x_eval[0,:] = (x[0,:]-X[0, 0])%(X[0, -1]-X[0, 0])+X[0, 0]
    
    # check if periodic in y
    if periodic[1]:
        
        x_eval[1,:] = (x[1,:]-Y[0, 0])%(Y[-1, 0]-Y[0, 0])+Y[0, 0]
        
    if periodic[2]:
        
        t = t%(time_data[0, -1]-time_data[0, 0])+time_data[0, 0]
    
    dt_data = time_data[0,1]-time_data[0,0]

    
    
    if linear:
        x_eval = x_eval[::-1,:] #change the order of x and y (Interpolant was defined for order x, y) 
        # Unsteady case
        if bool_unsteady:

            k = int((t-time_data[0, 0])/dt_data)
            # evaluate velocity field at time t_eval
            if k >= len(Interpolant_u)-1:
                u = Interpolant_u[-1](x_eval[1,:], x_eval[0,:])
                v = Interpolant_v[-1](x_eval[1,:], x_eval[0,:])
                
            else: 
        
                ui = Interpolant_u[k](x_eval[1,:], x_eval[0,:])
                uf = Interpolant_u[k+1](x_eval[1,:], x_eval[0,:])
                u = ((k+1)*dt_data-(t-time_data[0,0]))/dt_data*ui + ((t-time_data[0,0])-k*dt_data)/dt_data*uf
                

                vi = Interpolant_v[k](x_eval[1,:], x_eval[0,:])
                vf = Interpolant_v[k+1](x_eval[1,:], x_eval[0,:])
                v = ((k+1)*dt_data-(t-time_data[0,0]))/dt_data*vi + ((t-time_data[0,0])-k*dt_data)/dt_data*vf
            
        # Steady case        
        elif bool_unsteady == False:
                
            u = Interpolant_u(x_eval[1,:], x_eval[0,:])
            v = Interpolant_v(x_eval[1,:], x_eval[0,:])
            
    else:
        # Unsteady case
        if bool_unsteady:

            k = int((t-time_data[0, 0])/dt_data)
        
            # evaluate velocity field at time t_eval
            if k >= len(Interpolant_u)-1:
                
                u = Interpolant_u[-1](x_eval[1,:], x_eval[0,:], grid = False)
                v = Interpolant_v[-1](x_eval[1,:], x_eval[0,:], grid = False)
                
            else: 
        
                ui = Interpolant_u[k](x_eval[1,:], x_eval[0,:], grid = False)
                uf = Interpolant_u[k+1](x_eval[1,:], x_eval[0,:], grid = False)
                u = ((k+1)*dt_data-(t-time_data[0,0]))/dt_data*ui + ((t-time_data[0,0])-k*dt_data)/dt_data*uf

                vi = Interpolant_v[k](x_eval[1,:], x_eval[0,:], grid = False)
                vf = Interpolant_v[k+1](x_eval[1,:], x_eval[0,:], grid = False)
                v = ((k+1)*dt_data-(t-time_data[0,0]))/dt_data*vi + ((t-time_data[0,0])-k*dt_data)/dt_data*vf
            
        # Steady case        
        elif bool_unsteady == False:
                
            u = Interpolant_u(x_eval[1,:], x_eval[0,:], grid = False)
            v = Interpolant_v(x_eval[1,:], x_eval[0,:], grid = False)
        
    vel = np.array([u, v])
    
    return vel