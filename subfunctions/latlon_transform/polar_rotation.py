import numpy as np 

#Numerical error of 1e-5 when combining both functions for the longitude vector

def spherical_to_cartesian(latitude, longitude):
    from numpy import sin, cos, radians
    """
    Converts spherical coordinates (theta, phi) to Cartesian coordinates (x, y, z). Assume R constant
    
    Parameters:
        theta (float): Polar angle in radians (0 <= theta <= pi)
        phi (float): Azimuthal angle in radians (0 <= phi < 2*pi)
        
    Returns:
        tuple: Cartesian coordinates (x, y, z)
    """

    earthRadius = 6371*(10**3)

    theta = radians(90-latitude).ravel()
    phi = radians(longitude).ravel()
    
    x = (earthRadius * sin(theta) * cos(phi)).reshape(longitude.shape)
    y = (earthRadius * sin(theta) * sin(phi)).reshape(longitude.shape)
    z = (earthRadius * cos(theta)).reshape(longitude.shape)
    
    return x, y, z

def cartesian_to_spherical(x, y, z):  
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).
    
    Parameters:
        x (float): X-coordinate
        y (float): Y-coordinate
        z (float): Z-coordinate
        
    Returns:
        tuple: Spherical coordinates (theta, phi). Assume R constant
    """
    # Radial distance
    #r = math.sqrt(x**2 + y**2 + z**2)
    # Radial distance in the surface
    earthRadius = 6371*(10**3)
    
    # Polar angle (theta) from the top theta=0 in the pole 
    theta = np.degrees(np.acos(z/earthRadius))
    
    # Azimuthal angle (phi)
    phi = np.degrees(np.atan2(y, x))  # atan2 handles the correct quadrant for (y, x)
    
    return 90-theta, phi

def polar_rotation_rx(latitude,longitude,psi):

    from math import cos, sin, radians
    """
    Rotates a set of geographical coordinates (latitude and longitude) around the x-axis by a specified angle (psi).

    Parameters:
    - latitude: A numpy array representing the latitude values in degrees.
    - longitude: A numpy array representing the longitude values in degrees.
    - psi: The angle of rotation in radians. This angle specifies how much the coordinates should be rotated around the x-axis.

    Returns:
    - A tuple of two masked numpy arrays:
      - The first array contains the rotated latitude values.
      - The second array contains the rotated longitude values.
      Both arrays retain the mask from the input latitude and longitude arrays, ensuring that any masked values in the input are also masked in the output.

    Description:
    The function first constructs a rotation matrix `Rx` for rotating points around the x-axis by the angle `psi`. It then converts the input spherical coordinates (latitude and longitude) 
    into Cartesian coordinates. After applying the rotation matrix to these Cartesian coordinates, the function converts the rotated Cartesian coordinates back into spherical coordinates (latitude and longitude). 
    Finally, it returns the rotated coordinates as masked arrays, preserving any masks from the input arrays.
    """
    Rx = np.matrix([[ 1, 0         , 0          ],
                [ 0, cos(radians(psi)), -sin(radians(psi))],
                [ 0, sin(radians(psi)), cos(radians(psi))]])
    
    x, y, z = spherical_to_cartesian(latitude,longitude)
    rot_cartesian_matrix = Rx @ np.array([x.ravel(),y.ravel(),z.ravel()])
    lat_r, lon_r = cartesian_to_spherical(rot_cartesian_matrix[0,:].reshape(longitude.shape),rot_cartesian_matrix[1,:].reshape(longitude.shape),rot_cartesian_matrix[2,:].reshape(longitude.shape))
    #assign the mask
    return lat_r, lon_r