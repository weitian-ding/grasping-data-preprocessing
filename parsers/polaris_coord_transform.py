__author__ = 'srkiyengar'

import numpy as np
import math
import os
import logging.handlers


my_logger = logging.getLogger("pose_data_bulk_process")
LOG_LEVEL = logging.DEBUG


#the gripper handle has two locations (top and bottom)
#Static transformation is to convert measurements of a rigid body fro
#Pose and location when 449 is in the top - v1,q1 are 449 and v2,q2 are 339 at the center
v1_449 = [97.663788, -180.389755, -1895.446655]
q1_449 = [0.416817, -0.806037, 0.028007, -0.419267]
v2_339 = [78.019791, -26.525036, -1980.021118]
q2_339 = [0.222542, 0.551251, 0.281243, 0.753326]

#object_origin = [195.30,23.96,-1886.08]
#object_origin = [40.96,178.15,-1865.08]        #recorded using the new setup on 24 Feb. 2018
# object_origin = [44.28,178.75,1889.11]        # 17 March 2018 the -ve sign for z seems missing,noticed on May10,2018
#object_origin = [38.93, 180.00, -1886.60]      # recorded on 30 MArch 2018
object_origin = [39.63,181.75,-1887.68]         # May1 2018

#pose and location when v1,q1 of 339 is at the bottom and v2,q2 of 449 at the center
v1_339 = [203.19, -58.99, -1621.9]
q1_339 = [0.7765, -0.2614, -0.5724, 0.032]
v2_449 = [107.71, -127.45, -1699.52]
q2_449 = [0.2803, -0.5491, 0.4564, -0.6415]

labview_ndi_file = "../raw_data/305625-2018-02-19-20-50-26.txt-preprocessed"

def rotmat_to_axis_angle(R):

    r00 = R[0, 0]
    r01 = R[0, 1]
    r02 = R[0, 2]
    r10 = R[1, 0]
    r11 = R[1, 1]
    r12 = R[1, 2]
    r20 = R[2, 0]
    r21 = R[2, 1]
    r22 = R[2, 2]
    # catch the error
    angle = (r00 + r11 + r22 - 1) / 2

    theta = math.acos((r00 + r11 + r22 - 1) / 2)
    sinetheta = math.sin(theta)
    v = (2 * sinetheta) * theta

    cz = ((r10 - r01) / (2 * sinetheta)) * theta
    by = ((r02 - r20) / (2 * sinetheta)) * theta
    ax = ((r21 - r12) / (2 * sinetheta)) * theta

    return ax, by, cz

def axis_angle_to_rotmat(Rx,Ry,Rz):

    R = np.zeros((3,3),dtype=float)
    a1 = [Rx,Ry,Rz]
    angle = np.linalg.norm(a1)
    a1 = a1/angle

    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0-c

    R[0,0] = c + a1[0]*a1[0]*t
    R[1,1] = c + a1[1]*a1[1]*t
    R[2,2] = c + a1[2]*a1[2]*t

    tmp1 = a1[0]*a1[1]*t
    tmp2 = a1[2]*s
    R[1,0] = tmp1 + tmp2
    R[0,1] = tmp1 - tmp2

    tmp1 = a1[0]*a1[2]*t
    tmp2 = a1[1]*s
    R[2,0] = tmp1 - tmp2
    R[0,2] = tmp1 + tmp2

    tmp1 = a1[1]*a1[2]*t
    tmp2 = a1[0]*s
    R[2,1] = tmp1 + tmp2
    R[1,2] = tmp1 - tmp2

    return R

def rotation_matrix_from_quaternions(q_vector):

    '''
    :param q_vector: array, containing 4 values representing a unit quaternion that encodes rotation about a frame
    :return: an array of shape 3x3 containing the rotation matrix.
    Takes in array as [qr, qx, qy, qz]
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation, s = 1
    '''

    qr, qi, qj, qk = q_vector
    first = [1-2*(qj*qj+qk*qk), 2*(qi*qj-qk*qr),   2*(qi*qk+qj*qr)]
    second= [2*(qi*qj+qk*qr),   1-2*(qi*qi+qk*qk), 2*(qj*qk-qi*qr)]
    third = [2*(qi*qk-qj*qr),   2*(qj*qk+qi*qr),   1-2*(qi*qi+qj*qj)]
    R = np.array([first,second,third])
    return R


def homogenous_transform(R,vect):

    '''
    :param R: 3x3 matrix
    :param vect: list x,y,z
    :return:Homogenous transformation 4x4 matrix using R and vect
    '''

    H = np.zeros((4,4))
    H[0:3,0:3] = R
    frame_displacement = vect + [1]
    D = np.array(frame_displacement)
    D.shape = (1,4)
    H[:,3] = D
    return H

def inverse_homogenous_transform(H):

    '''
    :param H: Homogenous Transform Matrix
    :return: Inverse Homegenous Transform Matrix
    '''


    R = H[0:3,0:3]
    origin = H[:-1,3]
    origin.shape = (3,1)

    R = R.T
    origin = -R.dot(origin)
    return homogenous_transform(R,list(origin.flatten()))

def center_tool_339_to_gripper_center():

    '''
    The y-axis of 339 is aligned with the y axis of the gripper. The z-axis of the 339 will require a rotation of 90
    (counter clockwise with respect to y R (y,90) to get align gripper z axis to outward pointing. the origin of the
    339 needs to be moved in z-axis by + 40.45mm to get it to the origin of the gripper

    :return: homogenous transformation from 339 center to gripper center
    '''

    d =[0.0,0.0,40.45,1.0]
    H = np.zeros((4,4))
    H.shape = (4,4)
    H[:,3]= d
    H[(1,0),(1,2)]=1
    H[2,0]= -1
    return H

def center_tool_449_to_gripper_center():

    '''
    The y-axis of 4499 is aligned with the y axis of the gripper. The z-axis of the 449 will require a rotation of 90
    (counter clockwise with respect to y R (y,90) to get align gripper z axis to outward pointing. the origin of the
    449 needs to be moved in z-axis by + 35.36 (Not accurate) to get it to the origin of the gripper.
    the accurate measure from geometry 32.796

    :return: homogenous transformation from 339 center to gripper center
    '''

    d =[0.0,0.0,32.796,1.0]
    H = np.zeros((4,4))
    H.shape = (4,4)
    H[:,3]= d
    H[(1,0),(1,2)]=1
    H[2,0]= -1
    return H

def static_transform_449_top(q1,v1,q2,v2):
    '''

    :param q1: unit quaternions representing the rotation of the frame of 449 tool at the top
    :param v1: vector representing the rotation of the frame of 449 tool at the top
    :param q2: unit quaternions representing the rotation of the frame of 339 tool at the center
    :param v2: vector representing the rotation of the frame of 339 tool at the center
    :return: homogenous tranformation
    '''
    # H1 -  Homogenous transform from reference NDI frame to front tool
    # H2 -  Homogenous transform from reference NDI frame to center tool
    # H3 -  Homogenous transformation from the center tool frame to center of the gripper with axis rotated where the y
    # is parallel and between the two fingers and z is pointing outward


    R1 = rotation_matrix_from_quaternions(q1)
    H1 = homogenous_transform(R1, v1)
    h1 = inverse_homogenous_transform(H1)

    R2 = rotation_matrix_from_quaternions(q2)
    H2 = homogenous_transform(R2, v2)

    H3 = center_tool_339_to_gripper_center()
    H = (h1.dot(H2)).dot(H3)
    return H

def static_transform_339_bottom(q1,v1,q2,v2):
    '''

    :param q1: unit quaternions representing the rotation of the frame of 339 tool at the bottom of the gripper
    :param v1: vector representing the rotation of the frame of 339 tool
    :param q2: unit quaternions representing the rotation of the frame of 449 tool at the center
    :param v2: vector representing the rotation of the frame of 449
    :return: homogenous tranformation
    '''
    # H1 -  Homogenous transform from reference NDI frame to front tool
    # H2 -  Homogenous transform from reference NDI frame to center tool
    # H3 -  Homogenous transformation from the center tool frame to center of the gripper with axis rotated where the y
    # is parallel and between the two fingers and z is pointing outward


    R1 = rotation_matrix_from_quaternions(q1)
    H1 = homogenous_transform(R1, v1)
    h1 = inverse_homogenous_transform(H1)

    R2 = rotation_matrix_from_quaternions(q2)
    H2 = homogenous_transform(R2, v2)

    H3 = center_tool_449_to_gripper_center()
    H = (h1.dot(H2)).dot(H3)
    return H

# This static transform will move rotate from the NDI reference frame to object platform
def st_from_ndi_to_object_reference(position_vector):
    # the Z axis of the object is pointing upwards.
    # the x and y are marked. {for reference, +ive x aligned to the +z of the NDI reference


    first = [0,0,-1]
    second= [0,1,0]
    third = [1,0,0]
    R = np.array([first,second,third])
    H = homogenous_transform(R,position_vector)
    return H


# The Rx, Ry, Rz,x,y,z are the tcp position and pose when the object platform in the right orientation on the UR-5 table and
# TCP without any attachment at the origin of the platform. While aligning the TCP, the y of the TCP (opposite to the stub)
# should be aligned to the y of the platform.
#
def st_from_UR5_tcp_to_object_platform(Rx,Ry,Rz,x,y,z):

    first = [-1,0,0]
    second= [0,1,0]
    third = [0,0,-1]
    R = np.array([first,second,third])
    H = homogenous_transform(R,[0,0,0])
    R1 = axis_angle_to_rotmat(Rx,Ry,Rz)
    H1 = homogenous_transform(R1,x,y,z)
    # H1 represents Homogenous transformation from UR5 base to UR5 tool center point.
    # H represents Homogenous transformation from tool center point to object frame
    # F is the homogenous transformation from base to object frame
    F = np.dot(H1,H)
    return F


class ndi_transformation:
    def __init__(self,filename):

        try:
            with open(filename) as f:
                lines = f.readlines()
        except:
            self.error = "Unable to open the transformation.constants file"
            self.success = 0
            return

        val = []
        for line in lines:
            val.append(line)
        if len(val) != 9:
            self.error = "transformation.constants did not have 9 lines"
            self.success = 0
            return


        v1_449 = list(map(float,val[0].strip().split(",")[1:]))
        if len(v1_449) != 3:
            self.error = "transformation.constants- v1_449 is not 3 numbers"
            self.success = 0
            return

        q1_449 = list(map(float,val[1].strip().split(",")[1:]))
        if len(q1_449) != 4:
            self.error = "transformation.constants q1_449 is not 4 numbers"
            self.success = 0
            return

        v2_339 = list(map(float,val[2].strip().split(",")[1:]))
        if len(v2_339) != 3:
            self.error = "transformation.constants- v2_339 is not 3 numbers"
            self.success = 0
            return

        q2_339 = list(map(float,val[3].strip().split(",")[1:]))
        if len(q2_339) != 4:
            self.error = "transformation.constants q2_339 is not 4 numbers"
            self.success = 0
            return

        v1_339 = list(map(float,val[4].strip().split(",")[1:]))
        if len(v1_339) != 3:
            self.error = "transformation.constants- v1_339 is not 3 numbers"
            self.success = 0
            return

        q1_339 = list(map(float,val[5].strip().split(",")[1:]))
        if len(q1_339) != 4:
            self.error = "transformation.constants q1_339 is not 4 numbers"
            self.success = 0
            return

        v2_449 = list(map(float,val[6].strip().split(",")[1:]))
        if len(v2_449) != 3:
            self.error = "transformation.constants- v2_449 is not 3 numbers"
            self.success = 0
            return

        q2_449 = list(map(float,val[7].strip().split(",")[1:]))
        if len(q2_449) != 4:
            self.error = "transformation.constants q2_449 is not 4 numbers"
            self.success = 0
            return

        object_origin = list(map(float,val[8].strip().split(",")[1:]))
        if len(object_origin) != 3:
            self.error = "transformation.constants object_origin is not 3 numbers"
            self.success = 0
            return

        self.HT_from449_to_gripper_center = static_transform_449_top(q1_449,v1_449,q2_339,v2_339)
        self.HT_from339_to_gripper_center = static_transform_339_bottom(q1_339,v1_339,q2_449,v2_449)
        self.Inverse_HT_object = inverse_homogenous_transform(st_from_ndi_to_object_reference(object_origin))
        self.processed_lines = []
        self.fname = ""
        self.success = 1
        return

    @property
    def object_origin(self):
        return self.Inverse_HT_object

    # object origin is (x, y, z)
    @object_origin.setter
    def object_origin(self, object_origin):
        self.Inverse_HT_object = inverse_homogenous_transform(st_from_ndi_to_object_reference(object_origin))


class Transformer(object):
    def __init__(self, static_transform):
        self.st = static_transform

    def transform_single_example(self, tool1_params, tool2_params):
        # select the non-zero recording
        #    tool 1 => tool 449
        #    tool 2 => tool 339
        if (np.any(tool1_params)):
            x, y, z, qr, qi, qj, qk = tool1_params
        else:
            x, y, z, qr, qi, qj, qk = tool2_params

        R = rotation_matrix_from_quaternions([qr, qi, qj, qk])
        H = homogenous_transform(R, [x, y, z])

        if (np.any(tool2_params)):
            H = H.dot(self.st.HT_from339_to_gripper_center)
        else:
            H = H.dot(self.st.HT_from449_to_gripper_center)

        H_origin = self.st.Inverse_HT_object.dot(H)
        R_origin = H_origin[0:3,0:3]

        Rx,Ry,Rz = rotmat_to_axis_angle(R_origin)
        x = H_origin[0, 3]
        y = H_origin[1, 3]
        z = H_origin[2, 3]

        return (x, y, z, Rx, Ry, Rz)

