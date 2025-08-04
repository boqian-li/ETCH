import torch
import numpy as np
from vgtk.point3d import PointSet
# import vgtk.pc as pctk

class PointCloud():
	def __init__(self, xyz, feats=None):
		'''
		Args:
			xyz: [(b, )3, n]
			feats: [(b, )c, n]
		'''
		self._xyz = PointSet(xyz)
		self._feats = feats


	@property
	def data(self):
		return self._xyz, self._feats

# -- Temporary functions before transform.py is fixed -------------
def R_to_hom_np(pc, batch=False, rotate_only=False):
    pc_shape = pc.shape
    padding = 0 if rotate_only else 1
    if batch:
        ones = np.zeros((pc.shape[0], 1, pc.shape[2])) + padding
        return np.concatenate((pc, ones), axis=1)
    else:
        ones = np.zeros((1, pc.shape[1])) + padding
        return np.concatenate((pc, ones), axis=0)

def to_hom_np(pc, batch=False, rotate_only=False):
    pc_shape = pc.shape
    padding = 0 if rotate_only else 1
    if batch:
        ones = np.zeros((pc.shape[0], 1, pc.shape[2])) + padding
        return np.concatenate((pc, ones), axis=1)
    else:
        ones = np.zeros((1, pc.shape[1])) + padding
        return np.concatenate((pc, ones), axis=0)

def from_hom_np(pc, batch=False):
    if batch:
        return pc[..., :-1, :]
    else:
        return pc[:-1, :]

def transform_np(pc, T, batch=False):
    pc = pc.T

    if pc.shape[-2] == 3:
        pc = to_hom_np(pc)
    if batch:
        if T.ndim == 2:
            pcT =  np.einsum('pk,bkl->bpl', T, pc)
        elif T.ndim == 3:
            pcT = np.einsum('bpk,bkl->bpl', T, pc)
    else:
        pcT = np.matmul(T, pc)
    return from_hom_np(pcT).T

def cent(pc):
    axis = 0
    if pc.ndim == 2 and pc.shape[0] == 3:
        axis = 1
    elif pc.ndim == 3:
        axis += 1
        if pc.shape[1] ==3:
            axis += 1
    return pc - pc.mean(axis, keepdims=True)
