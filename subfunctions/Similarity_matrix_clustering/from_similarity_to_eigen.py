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

sys.path.append(parent_directory+"/subfunctions/trajectory_advection")
from Interpolant import generate_land_mask_interpolator 
sys.path.append(parent_directory+"/subfunctions/Parallelisation")
from parallelised_functions import split3D

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

def from_similarity_to_eigen_cut(Fmap, W_vec,e,K,k_exp,distance,land_mask,latitude,longitude):
    n=Fmap.shape[2]
    indices = np.tril_indices(n,0,n)
    print("The percentage of spercified elements is "+str(np.sum(W_vec < e)/np.sum(W_vec)))
    W_vec[W_vec < e] = 0

    # Create an empty matrix of zeros with shape (n, n)
    W = np.zeros((n, n))
    W[indices] = W_vec
    # Fill the upper triangular part 
    W = W + W.T - np.diag(np.diag(W))

    from scipy.ndimage import binary_dilation
    # Define the number of cells to expand the mask
    # Create a structuring element for dilation
    structuring_element = np.ones((2 * distance + 1, 2 * distance + 1))
    # Perform binary dilation to expand the mask
    expanded_land_mask = binary_dilation(land_mask, structure=structuring_element)
    exp_land_mask = generate_land_mask_interpolator(latitude,longitude,expanded_land_mask)
    Fmap_mask = exp_land_mask(Fmap[:,1,:],Fmap[:,0,:])
    IC_mask = np.sum(Fmap_mask,axis=0)
    IC_mask_final = np.where(IC_mask > 0)[0]
    W = np.delete(W, IC_mask_final, axis=0)  # Remove rows
    W = np.delete(W, IC_mask_final, axis=1)
    Fmap = np.delete(Fmap, IC_mask_final, axis=2)    
    print(str(IC_mask_final.shape)+" trajectories have been removed because they were too close to land or ending up too close to land")

    D=degree_matrix(W)
    indices_to_remove = np.where(D == K)[0]
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


def from_similarity_to_eigen_cut_zones(Fmap, W_vec,e,K,k_exp,distance,land_mask,latitude,longitude):
    n=Fmap.shape[2]
    indices = np.tril_indices(n,0,n)
    print("The percentage of spercified elements is "+str(np.sum(W_vec < e)/np.sum(W_vec)))
    W_vec[W_vec < e] = 0

    # Create an empty matrix of zeros with shape (n, n)
    W = np.zeros((n, n))
    W[indices] = W_vec
    # Fill the upper triangular part 
    W = W + W.T - np.diag(np.diag(W))

    from scipy.ndimage import binary_dilation
    # Define the number of cells to expand the mask
    # Create a structuring element for dilation
    structuring_element = np.ones((2 * distance + 1, 2 * distance + 1))
    # Perform binary dilation to expand the mask
    thick_land_mask = binary_dilation(land_mask, structure=structuring_element)
    from shapely.geometry import Point, Polygon
    # Define the vertices of the polygon
    polygon_vertices = [
        (-25,-110),
        (-10, -80),   # Vertex 2 (latitude, longitude)
        (-35, -50),    # Vertex 4 (latitude, longitude)
        (-40, -90)
    ]
    # Create a Polygon object using shapely
    polygon = Polygon(polygon_vertices)
    # Flatten the latitude and longitude matrices to create a list of points
    points = np.column_stack((latitude.ravel(), longitude.ravel()))
    # Use a list comprehension to check if each point is inside the polygon
    inside_mask = np.array([polygon.contains(Point(lat, lon)) for lat, lon in points])
    # Reshape the mask back to the original shape of the latitude matrix
    polygon_land_mask = inside_mask.reshape(latitude.shape)
    # Count the number of points inside the polygon
    inside_points_count = np.sum(polygon_land_mask)

    polygon_vertices = [
        (26,-92),
        (15, -98),   # Vertex 2 (latitude, longitude)
        (-20, -52),    # Vertex 4 (latitude, longitude)
        (-4, -54)
    ]
    # Create a Polygon object using shapely
    polygon = Polygon(polygon_vertices)
    # Flatten the latitude and longitude matrices to create a list of points
    points = np.column_stack((latitude.ravel(), longitude.ravel()))
    # Use a list comprehension to check if each point is inside the polygon
    inside_mask = np.array([polygon.contains(Point(lat, lon)) for lat, lon in points])
    # Reshape the mask back to the original shape of the latitude matrix
    polygon_land_mask = polygon_land_mask + inside_mask.reshape(latitude.shape)
    
    # Count the number of points inside the polygon
    inside_points_count = np.sum(polygon_land_mask)

    # Output the number of points inside the polygon
    print("Number of points inside the polygons:")
    print(str(inside_points_count))
    expanded_land_mask = thick_land_mask + polygon_land_mask
    exp_land_mask = generate_land_mask_interpolator(latitude,longitude,expanded_land_mask)
    print("Finally finished generating interpolator!")
    Fmap_mask = exp_land_mask(Fmap[:,1,:],Fmap[:,0,:])
    IC_mask = np.sum(Fmap_mask,axis=0)
    IC_mask_final = np.where(IC_mask > 0)[0]
    W = np.delete(W, IC_mask_final, axis=0)  # Remove rows
    W = np.delete(W, IC_mask_final, axis=1)
    Fmap = np.delete(Fmap, IC_mask_final, axis=2)    
    print(str(IC_mask_final.shape)+" trajectories have been removed because they were too close to land or ending up too close to land")

    D=degree_matrix(W)
    indices_to_remove = np.where(D == K)[0]
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


def cut_trajectories_in_W(Fmap, W_vec,distance,land_mask,latitude,longitude,eastSval=True, canadian_greenland=True):
    n=Fmap.shape[2]
    indices = np.tril_indices(n,0,n)

    # Create an empty matrix of zeros with shape (n, n)
    W = np.zeros((n, n))
    W[indices] = W_vec
    # Fill the upper triangular part 
    W = W + W.T - np.diag(np.diag(W))

    from scipy.ndimage import binary_dilation
    # Define the number of cells to expand the mask
    # Create a structuring element for dilation
    structuring_element = np.ones((2 * distance + 1, 2 * distance + 1))
    # Perform binary dilation to expand the mask
    thick_land_mask = binary_dilation(land_mask, structure=structuring_element)
    from shapely.geometry import Point, Polygon

        #tangent canadian archipelago leaving greenland tail
    polygon_vertices = [
        (-25,-110),
        (-5,-83),   # Vertex 2 (latitude, longitude)
        (-45, -50),    # Vertex 4 (latitude, longitude)
        (-45, -100)
    ]
    # Create a Polygon object using shapely
    polygon = Polygon(polygon_vertices)
    # Flatten the latitude and longitude matrices to create a list of points
    points = np.column_stack((latitude.ravel(), longitude.ravel()))
    # Use a list comprehension to check if each point is inside the polygon
    inside_mask = np.array([polygon.contains(Point(lat, lon)) for lat, lon in points])
    # Reshape the mask back to the original shape of the latitude matrix
    polygon_land_mask = inside_mask.reshape(latitude.shape)
    
    # Count the number of points inside the polygon
    inside_points_count = np.sum(polygon_land_mask)

    if canadian_greenland==True:
        # Define the vertices of the polygon
        #tangent canadian archipelago
        polygon_vertices = [
            (-20,-110),
            (10, -50),   # Vertex 2 (latitude, longitude)
            (-35, -50),    # Vertex 4 (latitude, longitude)
            (-40, -90)
        ]
        # Create a Polygon object using shapely
        polygon = Polygon(polygon_vertices)
        # Flatten the latitude and longitude matrices to create a list of points
        points = np.column_stack((latitude.ravel(), longitude.ravel()))
        # Use a list comprehension to check if each point is inside the polygon
        inside_mask = np.array([polygon.contains(Point(lat, lon)) for lat, lon in points])
        # Reshape the mask back to the original shape of the latitude matrix
        polygon_land_mask = polygon_land_mask + inside_mask.reshape(latitude.shape)
        # Count the number of points inside the polygon
        inside_points_count = np.sum(polygon_land_mask)
    if eastSval==True:
        #east of svalbard
        polygon_vertices = [
            (26,-70),
            (15, -98),   # Vertex 2 (latitude, longitude)
            (-20, -52),    # Vertex 4 (latitude, longitude)
            (-4, -54)
        ]
        # Create a Polygon object using shapely
        polygon = Polygon(polygon_vertices)
        # Flatten the latitude and longitude matrices to create a list of points
        points = np.column_stack((latitude.ravel(), longitude.ravel()))
        # Use a list comprehension to check if each point is inside the polygon
        inside_mask = np.array([polygon.contains(Point(lat, lon)) for lat, lon in points])
        # Reshape the mask back to the original shape of the latitude matrix
        polygon_land_mask = polygon_land_mask + inside_mask.reshape(latitude.shape)
        
        # Count the number of points inside the polygon
        inside_points_count = np.sum(polygon_land_mask)
    
    
    # Output the number of points inside the polygon
    print("Number of points inside the polygons:")
    print(str(inside_points_count))
    expanded_land_mask = thick_land_mask + polygon_land_mask
    exp_land_mask = generate_land_mask_interpolator(latitude,longitude,expanded_land_mask)
    print("Finally finished generating interpolator!")
    Fmap_mask = exp_land_mask(Fmap[:,1,:],Fmap[:,0,:])
    IC_mask = np.sum(Fmap_mask,axis=0)
    IC_mask_final = np.where(IC_mask > 0)[0]
    W = np.delete(W, IC_mask_final, axis=0)  # Remove rows
    W = np.delete(W, IC_mask_final, axis=1)

    Fmap = np.delete(Fmap, IC_mask_final, axis=2)    
    print(str(IC_mask_final.shape)+" trajectories have been removed because they were too close to land or ending up too close to land")
    return W, Fmap


def cut_trajectories_in_3W(Fmap, W_vec, w_disp, w_reweighted,distance,land_mask,latitude,longitude,eastSval=True, canadian_greenland=True):
    n=Fmap.shape[2]
    indices = np.tril_indices(n,0,n)

    # Create an empty matrix of zeros with shape (n, n)
    W = np.zeros((n, n))
    W[indices] = W_vec
    # Fill the upper triangular part 
    W = W + W.T - np.diag(np.diag(W))

    # Create an empty matrix of zeros with shape (n, n)
    W_disp = np.zeros((n, n))
    W_disp[indices] = w_disp
    # Fill the upper triangular part 
    W_disp = W_disp + W_disp.T - np.diag(np.diag(W_disp))

    # Create an empty matrix of zeros with shape (n, n)
    W_reweighted = np.zeros((n, n))
    W_reweighted[indices] = w_reweighted
    # Fill the upper triangular part 
    W_reweighted = W_reweighted + W_reweighted.T - np.diag(np.diag(W_reweighted))

    from scipy.ndimage import binary_dilation
    # Define the number of cells to expand the mask
    # Create a structuring element for dilation
    structuring_element = np.ones((2 * distance + 1, 2 * distance + 1))
    # Perform binary dilation to expand the mask
    thick_land_mask = binary_dilation(land_mask, structure=structuring_element)
    from shapely.geometry import Point, Polygon

        #tangent canadian archipelago leaving greenland tail
    polygon_vertices = [
        (-25,-110),
        (-5,-83),   # Vertex 2 (latitude, longitude)
        (-45, -50),    # Vertex 4 (latitude, longitude)
        (-45, -100)
    ]
    # Create a Polygon object using shapely
    polygon = Polygon(polygon_vertices)
    # Flatten the latitude and longitude matrices to create a list of points
    points = np.column_stack((latitude.ravel(), longitude.ravel()))
    # Use a list comprehension to check if each point is inside the polygon
    inside_mask = np.array([polygon.contains(Point(lat, lon)) for lat, lon in points])
    # Reshape the mask back to the original shape of the latitude matrix
    polygon_land_mask = inside_mask.reshape(latitude.shape)
    
    # Count the number of points inside the polygon
    inside_points_count = np.sum(polygon_land_mask)

    if canadian_greenland==True:
        # Define the vertices of the polygon
        #tangent canadian archipelago
        polygon_vertices = [
            (-20,-110),
            (10, -50),   # Vertex 2 (latitude, longitude)
            (-35, -50),    # Vertex 4 (latitude, longitude)
            (-40, -90)
        ]
        # Create a Polygon object using shapely
        polygon = Polygon(polygon_vertices)
        # Flatten the latitude and longitude matrices to create a list of points
        points = np.column_stack((latitude.ravel(), longitude.ravel()))
        # Use a list comprehension to check if each point is inside the polygon
        inside_mask = np.array([polygon.contains(Point(lat, lon)) for lat, lon in points])
        # Reshape the mask back to the original shape of the latitude matrix
        polygon_land_mask = polygon_land_mask + inside_mask.reshape(latitude.shape)
        # Count the number of points inside the polygon
        inside_points_count = np.sum(polygon_land_mask)
    if eastSval==True:
        #east of svalbard
        polygon_vertices = [
            (26,-70),
            (15, -98),   # Vertex 2 (latitude, longitude)
            (-20, -52),    # Vertex 4 (latitude, longitude)
            (-4, -54)
        ]
        # Create a Polygon object using shapely
        polygon = Polygon(polygon_vertices)
        # Flatten the latitude and longitude matrices to create a list of points
        points = np.column_stack((latitude.ravel(), longitude.ravel()))
        # Use a list comprehension to check if each point is inside the polygon
        inside_mask = np.array([polygon.contains(Point(lat, lon)) for lat, lon in points])
        # Reshape the mask back to the original shape of the latitude matrix
        polygon_land_mask = polygon_land_mask + inside_mask.reshape(latitude.shape)
        
        # Count the number of points inside the polygon
        inside_points_count = np.sum(polygon_land_mask)
    
    
    # Output the number of points inside the polygon
    print("Number of points inside the polygons:")
    print(str(inside_points_count))
    expanded_land_mask = thick_land_mask + polygon_land_mask
    exp_land_mask = generate_land_mask_interpolator(latitude,longitude,expanded_land_mask)
    print("Finally finished generating interpolator!")
    Fmap_mask = exp_land_mask(Fmap[:,1,:],Fmap[:,0,:])
    IC_mask = np.sum(Fmap_mask,axis=0)
    IC_mask_final = np.where(IC_mask > 0)[0]
    W = np.delete(W, IC_mask_final, axis=0)  # Remove rows
    W = np.delete(W, IC_mask_final, axis=1)

    W_disp = np.delete(W_disp, IC_mask_final, axis=0)  # Remove rows
    W_disp = np.delete(W_disp, IC_mask_final, axis=1)

    W_reweighted = np.delete(W_reweighted, IC_mask_final, axis=0)  # Remove rows
    W_reweighted = np.delete(W_reweighted, IC_mask_final, axis=1)

    Fmap = np.delete(Fmap, IC_mask_final, axis=2)    
    print(str(IC_mask_final.shape)+" trajectories have been removed because they were too close to land or ending up too close to land")
    return W, W_disp, W_reweighted, Fmap

def from_similarity_to_eigen_W(Fmap,d,W,K,k_exp):
    W_copy = W
    n=Fmap.shape[2]
    D=degree_matrix(W_copy)
    indices_to_remove = np.where(D == K)[0]
    print(str(indices_to_remove.shape)+" trajectories have been removed because they were not similar to any other trajectories")

    D = np.delete(np.delete(D, indices_to_remove, axis=0), indices_to_remove, axis=1)
    W_copy = np.delete(np.delete(W_copy, indices_to_remove, axis=0), indices_to_remove, axis=1)
    Fmap = np.delete(Fmap,indices_to_remove, axis=2)

    if d==10000:
        e=0
    else:
        e=1/d

    print("The percentage of spercified elements is "+str(np.sum(W_copy < e)/np.sum(W_copy.size)))
    W_copy[W_copy < e] = 0

    L=D-W_copy
    #Note that D, W, and therefore L, are real symmetric matrices (required for function "scipy.linalg.eigh")

    print("Computing first "+str(k_exp)+" eigenvalues")
    l = eigh(L,D,eigvals_only=False,subset_by_index=[0,k_exp-1])
    l_vect = l[1]
    l = l[0]

    return l_vect,l,Fmap#,k

def from_similarity_to_eigen_W_spard(Fmap,d,W,W_lyap,K,k_exp):
    W_lyap_copy = np.copy(W_lyap)
    Fmap_copy = np.copy(Fmap)
    n=Fmap.shape[2]

    if d==10000:
        print("No sparsification")

        D=degree_matrix(W_lyap_copy)
        indices_to_remove = np.where(D == K)[0]
        print(str(indices_to_remove.shape)+" trajectories have been removed because they were not similar to any other trajectories")

        D = np.delete(np.delete(D, indices_to_remove, axis=0), indices_to_remove, axis=1)
        W_lyap_copy = np.delete(np.delete(W_lyap_copy, indices_to_remove, axis=0), indices_to_remove, axis=1)
        Fmap_copy = np.delete(Fmap_copy,indices_to_remove, axis=2)

        L=D-W_lyap_copy
        #Note that D, W, and therefore L, are real symmetric matrices (required for function "scipy.linalg.eigh")

        print("Computing first "+str(k_exp)+" eigenvalues")
        l = eigh(L,D,eigvals_only=False,subset_by_index=[0,k_exp-1])
        l_vect = l[1]
        l = l[0]

    else:
        print("The percentage of spercified elements is "+str(np.sum(W < 1/d)/np.sum(W.size)))

        indices = np.where(W < 1/d)  
        W_lyap_copy[indices]=0

        D=degree_matrix(W_lyap_copy)
        indices_to_remove = np.where(D == K)[0]
        print(str(indices_to_remove.shape)+" trajectories have been removed because they were not similar to any other trajectories")

        D = np.delete(np.delete(D, indices_to_remove, axis=0), indices_to_remove, axis=1)
        W_lyap_copy = np.delete(np.delete(W_lyap_copy, indices_to_remove, axis=0), indices_to_remove, axis=1)
        Fmap_copy = np.delete(Fmap_copy,indices_to_remove, axis=2)

        L=D-W_lyap_copy
        #Note that D, W, and therefore L, are real symmetric matrices (required for function "scipy.linalg.eigh")

        print("Computing first "+str(k_exp)+" eigenvalues")
        l = eigh(L,D,eigvals_only=False,subset_by_index=[0,k_exp-1])
        l_vect = l[1]
        l = l[0]

    return l_vect,l,Fmap_copy#,k


def from_similarity_to_eigen_W_spard(Fmap,d,W,W_lyap,K,k_exp):
    W_lyap_copy = np.copy(W_lyap)
    Fmap_copy = np.copy(Fmap)
    n=Fmap.shape[2]

    if d==10000:
        print("No sparsification")

        D=degree_matrix(W_lyap_copy)
        indices_to_remove = np.where(D == K)[0]
        print(str(indices_to_remove.shape)+" trajectories have been removed because they were not similar to any other trajectories")

        D = np.delete(np.delete(D, indices_to_remove, axis=0), indices_to_remove, axis=1)
        W_lyap_copy = np.delete(np.delete(W_lyap_copy, indices_to_remove, axis=0), indices_to_remove, axis=1)
        Fmap_copy = np.delete(Fmap_copy,indices_to_remove, axis=2)

        L=D-W_lyap_copy
        #Note that D, W, and therefore L, are real symmetric matrices (required for function "scipy.linalg.eigh")

        print("Computing first "+str(k_exp)+" eigenvalues")
        l = eigh(L,D,eigvals_only=False,subset_by_index=[0,k_exp-1])
        l_vect = l[1]
        l = l[0]

    else:
        print("The percentage of spercified elements is "+str(np.sum(W < 1/d)/np.sum(W.size)))

        indices = np.where(W < 1/d)  
        W_lyap_copy[indices]=0

        D=degree_matrix(W_lyap_copy)
        indices_to_remove = np.where(D == K)[0]
        print(str(indices_to_remove.shape)+" trajectories have been removed because they were not similar to any other trajectories")

        D = np.delete(np.delete(D, indices_to_remove, axis=0), indices_to_remove, axis=1)
        W_lyap_copy = np.delete(np.delete(W_lyap_copy, indices_to_remove, axis=0), indices_to_remove, axis=1)
        Fmap_copy = np.delete(Fmap_copy,indices_to_remove, axis=2)

        L=D-W_lyap_copy
        #Note that D, W, and therefore L, are real symmetric matrices (required for function "scipy.linalg.eigh")

        print("Computing first "+str(k_exp)+" eigenvalues")
        l = eigh(L,D,eigvals_only=False,subset_by_index=[0,k_exp-1])
        l_vect = l[1]
        l = l[0]

    return l_vect,l,Fmap_copy, W_lyap_copy