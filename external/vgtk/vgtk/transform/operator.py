
import math
import numpy as np
import torch


'''
Point cloud transform:
    pc: 
        torch: [b, 3, p]
        np: [(b, )3, p]
'''


# def R_to_hom_np(R):
#     assert R.ndim == 2
#     R = np.concatenate((R,[[0,0,0]]),axis=0)
#     return np.concatenate((R,np.array([[0,0,0,1]]).T),axis=1)


# homogeneous transform with transformation matrix
# T: [(b, ){3/4}, 4] x [(b, ){3/4}, p]
# return: [(b, ){3/4}, p]
def transform(x, T, bnc=False):
    if bnc:
        x = x.permute(0,2,1).contiguous()

    assert x.shape[-2] == 4, 'homography required'
    if T.ndim == 2:
        x = torch.matmul(T, x)
    elif T.ndim == 3:
        x = torch.bmm(T, x)

    if bnc:
        x = x.permute(0,2,1).contiguous()
    return x

def transform_np(x, T, bnc=False, batch=False):
    if bnc:
        x = x.transpose(0, 2, 1)

    assert x.shape[-2] == 4, 'homography required'
    if batch:
        if T.ndim == 2:
            x =  np.einsum('pk,bkl->bpl', T, x)
        elif T.ndim == 3:
            x = np.einsum('bpk,bkl->bpl', T, x)
    else:
        x = np.matmul(T, x)

    if bnc:
        x = x.transpose(0, 2, 1)
    return x

# rotate 
# R [(b, )3, 3] / [(b, ){3/4}, 4] (hom)
# return: [(b, )3, p]
def rotate(x, R):
    if R.shape[-1] == 4:
        R = R[..., :3, :3].contiguous()
    if R.ndim == 2:
        x = torch.matmul(R, x)
    elif R.ndim == 3:
        x = torch.bmm(R, x)
    return x

def rotate_np(x, R, batch=False):
    if R.shape[-1] == 4:
        R = R[..., :3, :3]
    if batch:
        if R.ndim == 2:
            return np.einsum('pk,bkl->bpl', R, x)
        elif R.ndim == 3:
            return np.einsum('bpk,bkl->bpl', R, x)
    else:
        return np.matmul(R, x)

'''
euler angle [x,y,z] to rotation matrix [3,3]
'''    

def R_from_euler_np(angles):
    '''
    angles: [(b, )3]
    '''
    # R_x = np.array([[1,         0,                  0                   ],
    #                 [0,         math.cos(angles[0]), -math.sin(angles[0]) ],
    #                 [0,         math.sin(angles[0]), math.cos(angles[0])  ]
    #                 ])
    # R_y = np.array([[math.cos(angles[1]),    0,      math.sin(angles[1])  ],
    #                 [0,                     1,      0                   ],
    #                 [-math.sin(angles[1]),   0,      math.cos(angles[1])  ]
    #                 ])
                 
    # R_z = np.array([[math.cos(angles[2]),    -math.sin(angles[2]),    0],
    #                 [math.sin(angles[2]),    math.cos(angles[2]),     0],
    #                 [0,                     0,                      1]
    #                 ])
    # return np.dot(R_z, np.dot( R_y, R_x ))
    raise NotImplementedError()
        
# def rotate_from_euler(pc, angles, to_hom=False):
#     R = torch.from_numpy(R_from_euler_np(angles).astype(np.float32)).cuda()
#     pc = rotate(pc.transpose(1,2), R)
#     return pc.transpose(1,2)