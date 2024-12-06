import numpy as np 
from scipy.spatial.transform import Rotation as R

def _compute_dot_product(v1, v2):
    v1_normalized = v1 / np.linalg.norm(v1, axis=0)
    v2_normalized = v2 / np.linalg.norm(v2, axis=0)
    dot_product = np.sum(v1_normalized * v2_normalized, axis=0)
    
    return dot_product

def _compute_cross_product(v1, V2):
    v1_normalized = v1 / np.linalg.norm(v1, axis=0)
    v2_normalized = V2 / np.linalg.norm(V2, axis=0)
    cross_product = np.cross(v2_normalized.T, v1_normalized.T).T

    return cross_product

def compute_angle(v1, v2):
    
    dot_product = _compute_dot_product(v1, v2)
    cross_product = _compute_cross_product(v1, v2)
    
    angle = np.arctan2(cross_product, dot_product)
    return angle
    
def compute_rotation(v1, v2):
    """Rotation.
    
    Compute the rotation instance from Spipy.spatial.transform.Rotation, that transforms v1 to v2.

    Parameters
    ----------
    v1 : array_like
        Fisrt vector.
    v2 : array_like
        Second vector.

    Returns
    -------
    rotationon_instance : Rotation
        Rotation instance from Spipy.spatial.transform.Rotation.
    """
    
    cross_product = _compute_cross_product(v1, v2)

    ### Define the rotation axis and angle between the vectors
    rotation_axis = cross_product / np.linalg.norm(cross_product, axis=0)
    angle = compute_angle(v1, v2)
    
    ### Build the scipy Rotation instance
    print(angle.shape)
    print(rotation_axis.shape)
    print((angle * rotation_axis).shape )
    rotation_instance = R.from_rotvec((angle * rotation_axis).T)
    
    return rotation_instance  
        
def apply_rotation(v, rotation_instance):
    """Apply rotation.
    
    Apply the rotation instance to the vector v.

    Parameters
    ----------
    v : array_like
        Vector to rotate.
    rotation_instance : Rotation
        Rotation instance from Spipy.spatial.transform.Rotation.
        
    Returns
    -------
    rotated_vector : array_like
        Rotated vector.
    """
    
    ### Rotate the vector using the rotation instance
    rotated_vector = rotation_instance.apply(v)
    return rotated_vector.T
