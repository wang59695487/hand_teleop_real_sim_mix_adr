import numpy as np
from sapien.core import Pose

# CAM2ROBOT = Pose.from_transformation_matrix(np.array([[ 0.88277541 , 0.05624523, -0.46641617  ,1.16604067],
#                 [ 0.46973921, -0.09034254,  0.87817043 ,-0.95921005],
#                 [ 0.00725567, -0.99432123, -0.10617276  ,0.53628075],
#                 [ 0.         , 0.   ,       0.    ,      1.        ]]))
CAM2ROBOT = Pose.from_transformation_matrix(np.array([[ 0.05403586,  0.37436257, -0.92570664,  1.32308693],
 [ 0.99769317,  0.01790817,  0.06548009,  0.00584728],
 [ 0.04109101, -0.92710947, -0.3725313 ,  0.46607291],
 [ 0.        ,  0.        ,  0.        ,  1.        ]]))

#DESK2ROBOT_Z_AXIS = -0.17145
DESK2ROBOT_Z_AXIS = 0.00855

# simulation fov
fov = np.deg2rad(58.4)

# Relocate
RELOCATE_BOUND = [0.2, 0.8, -0.4, 0.4, DESK2ROBOT_Z_AXIS + 0.005, 0.6]

# TODO:
ROBOT2BASE = Pose(p=np.array([-0.55, 0., -DESK2ROBOT_Z_AXIS]))

# Table size
TABLE_XY_SIZE = np.array([0.6, 1.2])
TABLE_ORIGIN = np.array([0, -0.15])

# Robot table size
ROBOT_TABLE_XY_SIZE = np.array([0.76, 1.52])