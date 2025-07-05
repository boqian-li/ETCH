
import os
import random
import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as sciR

# from vgtk.pc.transform import *

'''
Point cloud augmentation
Only numpy function is included for now
'''

def R_from_euler_np(angles):
    '''
    angles: [(b, )3]
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(angles[0]), -math.sin(angles[0]) ],
                    [0,         math.sin(angles[0]), math.cos(angles[0])  ]
                    ])
    R_y = np.array([[math.cos(angles[1]),    0,      math.sin(angles[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(angles[1]),   0,      math.cos(angles[1])  ]
                    ])

    R_z = np.array([[math.cos(angles[2]),    -math.sin(angles[2]),    0],
                    [math.sin(angles[2]),    math.cos(angles[2]),     0],
                    [0,                     0,                      1]
                    ])
    return np.dot(R_z, np.dot( R_y, R_x ))


def rotate_point_cloud_90(data, normal = None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)

    rotation_angle = np.random.randint(low=0, high=4) * (np.pi/2.0)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)
    rotated_normal = np.dot(normal.reshape((-1, 3)), rotation_matrix) if normal is not None else None

    return rotated_data, rotated_normal, rotation_matrix


def rotate_point_cloud(data, R = None, max_degree = None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        R:
          3x3 array, optional Rotation matrix used to rotate the input
        max_degree:
          float, optional maximum DEGREE to randomly generate rotation
        Return:
          Nx3 array, rotated point clouds
    """
    # rotated_data = np.zeros(data.shape, dtype=np.float32)
    if R is not None:
      rotation_angle = R
    elif max_degree is not None:
      rotation_angle = np.random.randint(0, max_degree, 3) * np.pi / 180.0
    else:
      rotation_angle = sciR.random().as_matrix() if R is None else R

    if isinstance(rotation_angle, list) or  rotation_angle.ndim == 1:
      rotation_matrix = R_from_euler_np(rotation_angle)
    else:
      assert rotation_angle.shape[0] >= 3 and rotation_angle.shape[1] >= 3
      rotation_matrix = rotation_angle[:3, :3]

    if data is None:
      return None, rotation_matrix
    rotated_data = np.dot(rotation_matrix, data.reshape((-1, 3)).T)

    return rotated_data.T, rotation_matrix # return [N, 3],


def batch_rotate_point_cloud(data, R = None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original point clouds (torch tensor)
          R: numpy data
        Return:
          BxNx3 array, rotated point clouds
    """

    rotation_angle = sciR.random().as_matrix() if R is None else R

    if isinstance(rotation_angle, list) or rotation_angle.ndim == 1:
      rotation_matrix = R_from_euler_np(rotation_angle)
    else:
      assert rotation_angle.shape[0] >= 3 and rotation_angle.shape[1] >= 3
      rotation_matrix = rotation_angle[:3, :3]

    # since we are using pytorch...
    rotation_matrix = torch.from_numpy(rotation_matrix).to(data.device)
    rotation_matrix = rotation_matrix[None].repeat(data.shape[0],1,1)

    # Bx3x3, Bx3xN ->Bx3xN
    rotated_data = torch.matmul(rotation_matrix.double(), data.transpose(1,2).double())
    return rotated_data.transpose(1,2).contiguous().float(), rotation_matrix.float()


def rotate_point_cloud_with_normal(pc, surface_normal):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """

    rotation_angle = np.random.uniform() * 2 * np.pi
    # rotation_angle = np.random.randint(low=0, high=12) * (2*np.pi / 12.0)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    rotated_pc = np.dot(pc, rotation_matrix)
    rotated_surface_normal = np.dot(surface_normal, rotation_matrix)

    return rotated_pc, rotated_surface_normal


def rotate_perturbation_point_cloud(data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))

    rotated_data = np.dot(data, R)

    return rotated_data


def rotate_perturbation_point_cloud_with_normal_som(pc, surface_normal, som, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """

    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))

    rotated_pc = np.dot(pc, R)
    rotated_surface_normal = np.dot(surface_normal, R)
    rotated_som = np.dot(som, R)

    return rotated_pc, rotated_surface_normal, rotated_som


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, jittered point clouds
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data


from sklearn.neighbors import NearestNeighbors as nnbrs
def crop_point_cloud(data, k=0.05):
  N, C = data.shape

  crop_center = data[np.random.randint(N)]
  nbrs = nnbrs(n_neighbors=int(k*N)).fit(data)
  _, indices = nbrs.kneighbors(crop_center[None])
  cropped = np.delete(data, indices.flatten(), axis=0)
  return cropped

def permute(data):
  N, C = data.shape
  choice = np.random.choice(N,N,replace=True)
  return data[choice]
