"""Module including useful functions relative to rigid motion.

Functions:
    augment_matrix_coord: returns augmented vector
    get_rotation_mat_single_axis: computes rotation matrix around specificied axis (x,y or z)
    get_rigid_motion_mat_from_euler: computes 4X4 rigid transformation matrix, from the specified sequence of Euler/Cardan angles.   
    transform_rigid_motion: applies rigid transformation matrix to a given vector
    get_euler_zxy: retrieves ZXY Cardan sequene of angles from given rigid transformation matrix.
"""

####  PYTHON MODULES
import numpy as np



def augment_matrix_coord(array):

    n = len(array)
    return np.concatenate((array, np.ones((n,1))), axis = 1).T




def get_rotation_mat_single_axis( axis, angle ):

    """It computes the 3X3 rotation matrix relative to a single rotation of angle(rad) 
    about the axis(string 'x', 'y', 'z') for a righr handed CS"""

    if axis == 'x' : return np.array(([1,0,0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]))

    if axis == 'y' : return np.array(([np.cos(angle),0,np.sin(angle)],[0, 1, 0],[-np.sin(angle), 0, np.cos(angle)]))

    if axis == 'z' : return np.array(([np.cos(angle),-np.sin(angle),0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]))



def get_rigid_motion_mat_from_euler( alpha, axis_1, beta, axis_2, gamma, axis_3, t_x, t_y, t_z ):
    
    """It computes the 4X4 rigid motion matrix given a sequence of 3 Euler angles about the 3 axes 1,2,3 
    and the translation vector t_x, t_y, t_z"""

    rot1 = get_rotation_mat_single_axis( axis_1, alpha )
    rot2 = get_rotation_mat_single_axis( axis_2, beta )
    rot3 = get_rotation_mat_single_axis( axis_3, gamma )

    rot_mat = np.dot(rot1, np.dot(rot2,rot3))

    t = np.array(([t_x], [t_y], [t_z]))

    output = np.concatenate((rot_mat, t), axis = 1)

    return np.concatenate((output, np.array([[0.,0.,0.,1.]])), axis = 0)


def transform_rigid_motion( v, alpha, axis_1, beta, axis_2, gamma, axis_3, t_x, t_y, t_z ):

    """It transforms the NX3 array of points v according to the rigid transformation given by 
    the sequence of 3 Euler angles about the 3 axes 1,2,3 and the translation vector t_x, t_y, t_z 

    It returns a NX3 array of transformed points
    """ 

    rigid_motion_matrix = get_rigid_motion_mat_from_euler( alpha, axis_1, beta, axis_2, gamma, axis_3, t_x, t_y, t_z )

    return np.dot(rigid_motion_matrix, augment_matrix_coord(v))[0:3].T


def get_euler_zxy(rotation_matrix):

    """retrieves ZXY Cardan sequene of angles from given rigid transformation matrix."""

    if (rotation_matrix[2,1] != 1. and rotation_matrix[2,1] != -1.):

        euler_x_1 = np.arcsin(rotation_matrix[2,1])
        euler_y_1 = - np.arctan2( rotation_matrix[2,0]/np.cos(euler_x_1), rotation_matrix[2,2]/np.cos(euler_x_1))
        euler_z_1 = - np.arctan2( rotation_matrix[0,1]/np.cos(euler_x_1), rotation_matrix[1,1]/np.cos(euler_x_1))

        euler_x_2 = np.pi - np.arcsin(rotation_matrix[2,1])
        euler_y_2 = - np.arctan2( rotation_matrix[2,0]/np.cos(euler_x_1), rotation_matrix[2,2]/np.cos(euler_x_1))
        euler_z_2 = - np.arctan2( rotation_matrix[0,1]/np.cos(euler_x_1), rotation_matrix[1,1]/np.cos(euler_x_1))

        euler_1 = [np.rad2deg(euler_z_1), np.rad2deg(euler_x_1), np.rad2deg(euler_y_1)]
        euler_2 = [np.rad2deg(euler_z_2), np.rad2deg(euler_x_2), np.rad2deg(euler_y_2)]

    else:
        print('Gimbal Lock occurred')

    return euler_1, euler_2


if __name__ == "__main__":

    rotation_matrix = np.array([[0.9923,   -0.1226,    0.0170],
                       [0.0112,   -0.0483,   -0.9988],
                       [0.1233,    0.9913,   -0.0465]])

    e1, e2 = get_euler_zxy(rotation_matrix)

    print(e1)