import math
import numpy as np
import torch


# tranform to homogeneous coordinate
def to_hom(pc, rotate_only=False):
    pc_shape = pc.shape
    ones = torch.zeros(pc.shape[0], 1, pc.shape[2]).to(pc.device)
    if not rotate_only:
        ones = ones + 1
    return torch.cat((pc, ones), dim=1)

def to_hom_np(pc, batch=False, rotate_only=False):
    pc_shape = pc.shape
    padding = 0 if rotate_only else 1
    if batch:
        ones = np.zeros((pc.shape[0], 1, pc.shape[2])) + padding
        return np.concatenate((pc, ones), axis=1)
    else:
        ones = np.zeros((1, pc.shape[1])) + padding
        return np.concatenate((pc, ones), axis=0)

# transform to xyz coordinate
def from_hom(pc):
    return pc[..., :-1, :].contiguous()

def from_hom_np(pc, batch=False):
    if batch:
        return pc[..., :-1, :]
    else:
        return pc[:-1, :]
        