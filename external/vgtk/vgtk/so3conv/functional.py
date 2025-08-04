import math
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm

# from utils_cuda import _neighbor_query, _spherical_conv
dirname = os.path.dirname(__file__)


def bp():
    import pdb;pdb.set_trace()

import vgtk
from vgtk import pc
import epn_grouping as cuda_nn



def acos_safe(x, eps=1e-4):
    sign = torch.sign(x)
    slope = np.arccos(1-eps) / eps
    return torch.where(abs(x) <= 1-eps,
                    torch.acos(x),
                    torch.acos(sign * (1 - eps)) - slope*sign*(abs(x) - 1 + eps))

def anchor_knn(a_src, a_tgt, k=3, metric="spherical"):
    '''
    for each anchor in a_tgt, find k nearest neighbors in a_src
        ax3, ax3 -> axk indices, axk distances
    '''
    a_src = a_src.unsqueeze(0)
    a_tgt = a_tgt.unsqueeze(1)
    # sum(a_tgt x k)
    if metric == 'spherical':
        dists = torch.sum(a_src*a_tgt, dim=2) - 1.0
        val, idx = dists.topk(k=k,dim=1, largest=True)
    elif metric == 'angular':
        dists = acos_safe(torch.sum(a_src*a_tgt, dim=2))
        # dists[dists != dists] = np.pi
        val, idx = dists.topk(k=k,dim=1, largest=False)
    else:
        dists = torch.sum((a_src - a_tgt)**2, dim=2)
        val, idx = dists.topk(k=k,dim=1, largest=False)
    return val, idx

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def inter_so3conv_feat_grouping(inter_idx, inter_w, feats):
    b, p, pnn = inter_idx.shape
    _, c, q, a = feats.shape
    device = inter_idx.device
    new_feats = batched_index_select(feats, 2, inter_idx.long().view(b, -1)).view(b, -1, p, pnn, a)
    new_feats = torch.einsum('bcpna,bpakn->bckpa', new_feats, inter_w).contiguous()
    return new_feats

# pc: [nb,np,3] -> feature: [nb,1,np,na]
def get_occupancy_features(pc, n_anchor, use_center=False):
    nb, np, nd = pc.shape
    has_normals = nd == 6

    features = torch.zeros(nb, 1, np, n_anchor) + 1
    features = features.float().to(pc.device)

    if has_normals:
        ns = pc[:,:,3:]
        if n_anchor > 1:
            anchors = torch.from_numpy(get_anchors())
            features_n = torch.einsum('bni,aij->bjna',ns.anchors)
        else:
            features_n = ns.transpose(1,2)[...,None].contiguous()
        features = torch.cat([features,features_n],1)

    if use_center:
        features[:,:,0,:] = 0.0

    return features

# Add shadow xyz (inf)
# [b, c, n] -> [b, c, n+1]
def add_shadow_point(x):
    b, c, _ = x.shape
    shadow_point = torch.ones(b,c,1).float().to(x.device) * 1e4
    x = torch.cat((x,shadow_point), dim=2).contiguous()
    return x

# Add shadow feature (zeros)
# [b, c, n, a] -> [b, c, n+1, a]
def add_shadow_feature(x):
    b, c, _, a = x.shape
    shadow_point = torch.zeros(b,c,1,a).float().to(x.device)
    x = torch.cat((x,shadow_point), dim=2).contiguous()
    return x

# (x,y,z) points derived from conic parameterization
# def get_kernel_points_np(radius, aperature, kernel_size, multiplier=1):
#     assert isinstance(kernel_size, int)
#     rrange = np.linspace(0, radius, kernel_size, dtype=np.float32)
#     kps = []

#     for ridx, ri in enumerate(rrange):
#         alpharange = zpconv.get_angular_kernel_points_np(aperature, ridx * multiplier + 1)
#         for aidx, alpha in enumerate(alpharange):
#             r_r = ri * np.tan(alpha)
#             thetarange = np.linspace(0, 2 * np.pi, aidx * 2 + 1, endpoint=False, dtype=np.float32)
#             xs = r_r * np.cos(thetarange)
#             ys = r_r * np.sin(thetarange)
#             zs = np.repeat(ri, aidx * 2 + 1)
#             kps.append(np.vstack([xs,ys,zs]).T)

#     kps = np.vstack(kps)
#     return kps

# def get_spherical_kernel_points_np(radius, kernel_size, multiplier=3):
#     assert isinstance(kernel_size, int)
#     rrange = np.linspace(0, radius, kernel_size, dtype=np.float32)
#     kps = []

#     for ridx, r_i in enumerate(rrange):
#         asize = ridx * multiplier + 1
#         bsize = ridx * multiplier + 1
#         alpharange = np.linspace(0, 2*np.pi, asize, endpoint=False, dtype=np.float32)
#         betarange = np.linspace(0, np.pi, bsize, endpoint=True, dtype=np.float32)

#         xs = r_i * np.cos(alpharange[:, None]) * np.sin(betarange[None])
#         ys = r_i * np.sin(alpharange[:, None]) * np.sin(betarange[None])

#         zs = r_i * np.cos(betarange)[None].repeat(asize, axis=0)
#         kps.append(np.vstack([xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)]).T)

#     kps = np.vstack(kps)
#     return kps

def get_sphereical_kernel_points_from_ply(radius, kernel_size):
    assert kernel_size <= 3 and kernel_size > 0
    mapping = {1:24, 2:30, 3:66} # TODO
    root = vgtk.__path__[0]
    anchors_path = os.path.join(root, 'data', 'anchors')
    ply_path = os.path.join(anchors_path, f'kpsphere{mapping[kernel_size]:d}.ply')
    # kp24
    ply = pc.load_ply(ply_path).astype('float32')
    def normalize(pc, radius):
        r = np.sqrt((pc**2).sum(1).max())
        return pc*radius/r
    return normalize(ply, radius)

def ball_query(query_points, support_points, radius, n_sample, support_feats=None):
    idx = pc.ball_query_index(query_points, support_points, radius, n_sample)
    support_points = add_shadow_point(support_points)

    if support_feats is None:
        return idx, pc.group_nd(support_points, idx)
    else:
        return idx, pc.group_nd(support_points, idx), pc.group_nd(support_feats, idx)

# initial_anchor_query(
#     at::Tensor centers, //[b, 3, nc]
#     at::Tensor xyz,  //[m, 3]
#     at::Tensor kernel_points, // [ks, na, 3]
#     const float radius, const float sigma)
def initial_anchor_query(frag, centers, kernels, r, sigma):
    return cuda_nn.initial_anchor_query(centers, frag, kernels, r, sigma)

def inter_spconv_grouping_ball(xyz, stride, radius, n_neighbor, lazy_sample=True):

    n_sample = math.ceil(xyz.shape[2] / stride)
    # [b, 3, p1] x [p2] -> [b,p2] x [b, 3, p2]
    idx, sample_xyz = pc.furthest_sample(xyz, n_sample, lazy_sample)
    # [b, p2, nn]
    ball_idx, grouped_xyz = ball_query(sample_xyz, xyz, radius, n_neighbor)
    # [b, 3, p1+1] x [b, p2, nn] -> [b, 3, p2, nn]
    grouped_xyz = grouped_xyz - sample_xyz.unsqueeze(3)
    return grouped_xyz, ball_idx, idx, sample_xyz


def inter_spconv_grouping_naive(inter_idx, inter_w, feats):
    b, p, pnn = inter_idx.shape
    _, c, q, a = feats.shape
    device = inter_idx.device
    new_feats = batched_index_select(feats, 2, inter_idx.long().view(b, -1)).view(b, -1, p, pnn, a)
    new_feats = torch.einsum('bcpna,bpakn->bckpa', new_feats, inter_w).contiguous()
    return new_feats


def inter_pooling_naive(inter_idx, sample_idx, feats, alpha=0.5):
    b, p, pnn = inter_idx.shape
    _, c, q, a = feats.shape

    new_feats = batched_index_select(feats, 2, sample_idx.long())
    grouped_feats = batched_index_select(add_shadow_feature(feats), 2, inter_idx.long().view(b, -1)).view(b, -1, p, pnn, a)
    return alpha * new_feats + (1 - alpha) * grouped_feats.mean(3)


def inter_blurring_naive(inter_idx, feats, alpha=0.5):
    b, p, pnn = inter_idx.shape
    _, c, q, a = feats.shape
    assert p == q
    grouped_feats = batched_index_select(add_shadow_feature(feats), 2, inter_idx.long().view(b, -1)).view(b, -1, p, pnn, a)
    return alpha * feats + (1 - alpha) * grouped_feats.mean(3)


def inter_so3conv_blurring(xyz, feats, n_neighbor, radius, stride,
                           inter_idx=None, lazy_sample=True, radius_expansion=1.0):
    if inter_idx is None:
        _, inter_idx, sample_idx, sample_xyz = inter_spconv_grouping_ball(xyz, stride, radius * radius_expansion, n_neighbor, lazy_sample)

    if stride == 1:
        return inter_blurring_naive(inter_idx, feats), xyz
    else:
        return inter_pooling_naive(inter_idx, sample_idx, feats), sample_xyz

def inter_so3conv_grouping(xyz, feats, stride, n_neighbor,
                          anchors, kernels, radius, sigma,
                          inter_idx=None, inter_w=None, lazy_sample=True,
                          radius_expansion=1.0, pooling=None):
    '''
        xyz: [nb, 3, p1] coordinates
        feats: [nb, c_in, p1, na] features
        anchors: [na, 3, 3] rotation matrices
        kernels: [ks, 3] kernel points
        inter_idx: [nb, p2, nn] grouped points, where p2 = p1 / stride
        inter_w: [nb, p2, na, ks, nn] kernel weights:
                    Influences of each neighbor points on each kernel points
                    under the respective SO3 rotations
    '''

    if pooling is not None and stride > 1 and feats.shape[1] > 1:
        # Apply low pass blurring before strided conv
        if pooling == 'stride':
            # NOTE: if meanpool replaces stride, nn and radius needs to be matched with the next conv
            pool_stride = stride
            # TODO: REMOVE HARD CODING
            stride_nn = int(n_neighbor * pool_stride**0.5)
            stride = 1
        elif pooling == 'no-stride':
            pool_stride = 1
            stride_nn = n_neighbor
        else:
            raise NotImplementedError(f"Pooling mode {pooling} is not implemented!")

        feats, xyz = inter_so3conv_blurring(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None

    if inter_idx is None:
        grouped_xyz, inter_idx, sample_idx, new_xyz = inter_spconv_grouping_ball(xyz, stride,
                                                                         radius * radius_expansion, n_neighbor, lazy_sample)
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)


        #####################DEBUGDEBUGDEBUGDEBUG####################################
        # print(xyz.shape)
        # xyz_sample = (xyz - xyz.mean(2, keepdim=True))[0]
        # gsample1 = xyz_sample[:,inter_idx[0,12].long()]
        # gsample2 = xyz_sample[:,inter_idx[0,25].long()]
        # gsample3 = xyz_sample[:,inter_idx[0,31].long()]
        # pc.save_ply('data/gsample2.ply', gsample2.T.cpu().numpy(), c='r')
        # pc.save_ply('data/gsample3.ply', gsample3.T.cpu().numpy(), c='r')
        # pc.save_ply('data/xyz.ply', xyz_sample.T.cpu().numpy())

        # for bi in range(new_xyz.shape[0]):
        #     pc.save_ply(f'vis/gsample{bi}.ply', new_xyz[bi].T.cpu().numpy())
        # # import ipdb; ipdb.set_trace()
        #############################################################################
    else:
        sample_idx = None
        new_xyz = xyz

    feats = add_shadow_feature(feats)

    new_feats = inter_so3conv_feat_grouping(inter_idx, inter_w, feats) # [nb, c_in, ks, np, na]

    return inter_idx, inter_w, new_xyz, new_feats, sample_idx

def inter_so3conv_grouping_anchor(grouped_xyz, anchors,
                                  kernels, sigma, interpolate='linear'):
    '''
        grouped_xyz: [b, 3, p2, nn]
        ball_idx: [b, p2, nn]
        anchors: [na, 3, 3]
        sample_idx: [b, p2]
    '''

    # kernel rotations:  3, na, ks
    rotated_kernels = torch.matmul(anchors, kernels.transpose(0,1)).permute(1,0,2).contiguous()

    # calculate influences: [3, na, ks] x [b, 3, p2, nn] -> [b, p2, na, ks, nn] weights
    t_rkernels = rotated_kernels[None, :, None, :, :, None]
    t_gxyz = grouped_xyz[...,None,None,:]

    if interpolate == 'linear':
        #  b, p2, na, ks, nn
        dists = torch.sum((t_gxyz - t_rkernels)**2, dim=1)
        # dists = torch.sqrt(torch.sum((t_gxyz - t_rkernels)**2, dim=1))
        inter_w = F.relu(1.0 - dists/sigma, inplace=True)
        # inter_w = F.relu(1.0 - dists / (3*sigma)**0.5, inplace=True)

        # torch.set_printoptions(precision=2, sci_mode=False)
        # print('---sigma----')
        # print(sigma)
        # print('---mean distance---')
        # print(dists.mean())
        # print(dists[0,10,0,6])
        # print('---weights---')
        # print(inter_w[0,10,0,6])
        # print('---summary---')
        # summary = torch.sum(inter_w[0,:,0] > 0.1, -1)
        # print(summary.float().mean(0))
        # import ipdb; ipdb.set_trace()
    else:
        raise NotImplementedError("kernel function %s is not implemented!"%interpolate)

    return inter_w

def get_intra_kernels(aperature, kernel_size):
    kernels = np.linspace(0, 0.5*aperature, kernel_size, dtype=np.float32)
    kernels = torch.from_numpy(kernels)
    return kernels

def intra_so3conv_grouping(intra_idx, feature):
    '''
        intra_idx: [na,pnn] so3 neighbors
        feature: [nb, c_in, np, na] features
    '''

    # group features -> [nb, c_in, pnn, np, na]

    nb, c_in, nq, na = feature.shape
    _, pnn = intra_idx.shape

    feature1 = feature.index_select(3, intra_idx.view(-1)).view(nb, c_in, nq, na, pnn)
    grouped_feat =  feature1.permute([0,1,4,2,3]).contiguous()

    # print(torch.sort(grouped_feat[0,0].mean(0)))
    # print(torch.sort(grouped_feat[1,0].mean(0)))
    # print(torch.sort(grouped_feat[0,0,0]))
    # print(torch.sort(grouped_feat[1,0,0]))
    # print(torch.sort(grouped_feat[0,0,1]))
    # print(torch.sort(grouped_feat[1,0,1]))

    # def find_rel(idx1, idx2):
    #     idx1 = idx1.cpu().numpy().squeeze()
    #     idx2 = idx2.cpu().numpy().squeeze()
    #     idx3 = np.zeros_like(idx1)
    #     for i in range(len(idx1)):
    #         idx3[idx2[i]] = idx1[i]
    #     return idx3

    # def comb_rel(idx1, idx2):
    #     idx1 = idx1.cpu().numpy().squeeze()
    #     idx2 = idx2.cpu().numpy().squeeze()
    #     idx3 = np.zeros_like(idx1)
    #     for i in range(len(idx1)):
    #         idx3[i] = idx1[idx2[i]]
    #     return idx3

    # rel01 = find_rel(torch.sort(grouped_feat[0,0,0])[1], torch.sort(grouped_feat[0,0,1])[1])
    # rel02 = find_rel(torch.sort(grouped_feat[0,0,0])[1], torch.sort(grouped_feat[1,0,0])[1])
    # rel13 = find_rel(torch.sort(grouped_feat[0,0,1])[1], torch.sort(grouped_feat[1,0,1])[1])
    # rel23 = find_rel(torch.sort(grouped_feat[1,0,0])[1], torch.sort(grouped_feat[1,0,1])[1])

    # rel_in = find_rel(torch.sort(feature[0,0,0])[1], torch.sort(feature[1,0,0])[1])
    # rel_out = find_rel(torch.sort(grouped_feat[0,0,0])[1], torch.sort(grouped_feat[1,0,0])[1])

    # import ipdb; ipdb.set_trace()

    return grouped_feat

# initialize so3 sampling
import vgtk.functional as fr
import os

GAMMA_SIZE = 3
ROOT = vgtk.__path__[0]
ANCHOR_PATH = os.path.join(ROOT, 'data', 'anchors/sphere12.ply')
Rs, R_idx, canonical_relative = fr.icosahedron_so3_trimesh(ANCHOR_PATH, GAMMA_SIZE)


def select_anchor(anchors, k):
    if k == 1:
        return anchors[29][None]
    elif k == 20:
        return anchors[::3]
    elif k == 40:
        return anchors.reshape(20,3,3,3)[:,:2].reshape(-1,3,3)
    else:
        return anchors


def get_anchors(k=60):
    return select_anchor(Rs,k)

def get_intra_idx():
    return R_idx

def get_canonical_relative():
    return canonical_relative
