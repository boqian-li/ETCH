import math
import os
import numpy as np
import time
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import vgtk.pc as pctk
from . import functional as L

# Basic ZPConv
# [b, c1, k, p, a] -> [b, c2, p, a]
class BasicZPConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, debug=False):
        super(BasicZPConv, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size

        # TODO: initialization argument
        # TODO: add bias

        if debug:
            W = torch.zeros(self.dim_out, self.dim_in*self.kernel_size) + 1
            self.register_buffer('W', W)
        else:
            W = torch.empty(self.dim_out, self.dim_in, self.kernel_size)
            # nn.init.xavier_normal_(W, gain=0.001)
            nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('relu'))
            # nn.init.normal_(W, mean=0.0, std=0.3)
            W = W.view(self.dim_out, self.dim_in*self.kernel_size)

            self.register_parameter('W', nn.Parameter(W))
            bias = torch.zeros(self.dim_out) + 1e-3
            bias = bias.view(1,self.dim_out,1)
            self.register_parameter('bias', nn.Parameter(bias))

        #self.W = nn.Parameter(torch.Tensor(self.dim_out, self.dim_in*self.kernel_size))

    def forward(self, x):
        bs, np, na = x.shape[0], x.shape[3], x.shape[4]
        x = x.view(bs, self.dim_in*self.kernel_size, np*na)
        x = torch.matmul(self.W, x)
        x = x + self.bias
        x = x.view(bs, self.dim_out, np, na)
        return x


# A single Intra ZPConv
# [b, c1, p, a_in] -> [b, c1, k, p, a_out] -> [b, c2, p, a_out]
class IntraZPConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size,
                 aperture, sigma,
                 anchor_nn, anchor_in, anchor_out=None):
        super(IntraZPConv, self).__init__()

        if anchor_out is None:
            anchor_out = anchor_in

        anchor_in = L.get_anchors(anchor_in)
        anchor_out = L.get_anchors(anchor_out)
        kernels = L.get_intra_kernels(aperture, kernel_size)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernels.shape[0]
        self.basic_conv = BasicZPConv(dim_in, dim_out, self.kernel_size)

        self.aperture = aperture
        self.sigma = sigma
        self.anchor_nn = anchor_nn

        intra_idx, intra_w = L.get_intra_kernel_weights(anchor_in, anchor_out,
                                                        kernels, self.anchor_nn,
                                                        self.aperture, self.sigma)

        # self.register_buffer('anchor_in', anchor_in)
        self.register_buffer('anchor_out', anchor_out)
        self.register_buffer('kernels', kernels)
        self.register_buffer('intra_idx', intra_idx)
        self.register_buffer('intra_w', intra_w)

    def forward(self, x):
        feats = L.intra_zpconv_grouping_naive(self.intra_idx, self.intra_w, x.feats)
        # import ipdb; ipdb.set_trace()
        # feats = feats / torch.sin(self.kernels)[:, None, None]
        feats = self.basic_conv(feats)
        return SphericalPointCloud(x.xyz, feats, self.anchor_out)

# A single Inter ZPConv
# [b, c1, p1, a] -> [b, c1, k, p2, a] -> [b, c2, p2, a]
class InterZPConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride,
                 radius, aperture, sigma,
                 anchors_dim, n_neighbor, anchor_nn, multiplier=3, lazy_sample=True):
        super(InterZPConv, self).__init__()

        anchors = L.get_anchors(anchors_dim)

        kernels = L.get_kernel_rings_np(radius, aperture, kernel_size, multiplier=multiplier)

        # kernels = L.get_kernel_points_np(radius, aperture, kernel_size - 2, multiplier=multiplier)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernels.shape[0]
        self.stride = stride
        self.basic_conv = BasicZPConv(dim_in, dim_out, self.kernel_size)

        self.radius = radius
        self.aperture = aperture
        self.sigma = sigma
        self.n_neighbor = n_neighbor
        self.anchor_nn = anchor_nn
        self.lazy_sample = lazy_sample

        self.register_buffer('anchors', anchors)
        self.register_buffer('kernels', torch.from_numpy(kernels))

    def forward(self, x, inter_idx=None, inter_w=None):
        # TODO: simplify lazy_sample
        inter_idx, inter_w, xyz, feats = \
            L.inter_zpconv_grouping(x.xyz, x.feats, self.stride, self.n_neighbor,
                                  self.anchors, self.kernels, self.anchor_nn,
                                  self.radius, self.aperture, self.sigma,
                                  inter_idx, inter_w, self.lazy_sample)
        feats = self.basic_conv(feats)
        return inter_idx, inter_w, SphericalPointCloud(xyz, feats, self.anchors)

# [b, c, p, a1] -> [b, c, p, a2]
class AnchorProp(nn.Module):
    def __init__(self, anchor_in, anchor_out, sigma, k=6):
        super(AnchorProp, self).__init__()

        anchor_in = L.get_anchors(anchor_in)
        anchor_out = L.get_anchors(anchor_out)
        idx, w = L.compute_anchor_weights(anchor_in, anchor_out, k=k, sigma=sigma)
        self.sigma = sigma

        self.register_buffer('anchor_out', anchor_out)
        self.register_buffer('idx', idx)
        self.register_buffer('w', w)

    def forward(self, x):
        feats = L.anchor_prop(x.feats, self.idx, self.w)
        return SphericalPointCloud(x.xyz, feats, self.anchor_out)
