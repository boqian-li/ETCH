import time
import numpy as np
import torch

import epn_grouping as cuda_nn
import epn_gathering as cuda_gather

def batch_gather(x, idx, dim=1):
    x = cuda_gather.gather_points_forward(x, idx.int())
    return x

'''
This file contains operators on point cloud that
depend on indexing.
The index is also returned by all functions
'''

# uniformly random resample
# pc: [p, 3], n_sample: int
def uniform_resample_index_np(pc, n_sample, batch=False):
    if batch == True:
        raise NotImplementedError('resample in batch is not implemented')
    n_point = pc.shape[0]
    if n_point >= n_sample:
        # downsample
        idx = np.random.choice(n_point, n_sample, replace=False)
    else:
        # upsample
        idx = np.random.choice(n_point, n_sample-n_point, replace=True)
        idx = np.concatenate((np.arange(n_point), idx), axis=0)
    return idx

def uniform_resample_np(pc, n_sample, label=None, batch=False):
    if batch == True:
        raise NotImplementedError('resample in batch is not implemented')
    idx = uniform_resample_index_np(pc, n_sample, batch)
    if label is None:
        return idx, pc[idx]
    else:
        return idx, pc[idx], label[idx]


# nearest neighbor
def knn_index_np(pc, k, batch=False):
    raise NotImplementedError('knn is not implemented')


# gather points with index
# [b, c, n] x [b, m1(, m2, ...)] -> [b, c, m1(, m2, ...)]
def group_nd(pc, idx):
    b = idx.shape[0]
    pc = batch_gather(pc, idx.view(b, -1).contiguous(), dim=2)
    pc = pc.view(b, -1, *idx.shape[1:])
    return pc

# ball query
# [b, 3, n] x [b, 3, m] x r x k -> [b, n, k]
def ball_query_index(query_points, support_points, radius, n_sample):
    # TODO remove permute
    # query_points = query_points.permute(0,2,1).contiguous()
    # support_points = support_points.permute(0,2,1).contiguous()
    if not hasattr(ball_query_index, "call_count"):
        ball_query_index.call_count = 0

    # 仅在前 N 次调用时打印
    N = 16
    if ball_query_index.call_count < N:
        print("query_points.shape: {}, support_points.shape: {}, radius: {}, n_sample: {}".format(query_points.shape, support_points.shape, radius, n_sample))
        ball_query_index.call_count += 1
    idx = cuda_nn.ball_query(query_points, support_points, radius, n_sample)
    return idx


# [b, 3, n] x [m] -> [b, m]
def furthest_sample_index(pc, n_sample, lazy_sample):
    if pc.shape[2] == n_sample or lazy_sample:
        nb = pc.shape[0]
        rst = torch.arange(n_sample).view(1, -1).expand(nb,-1).int().contiguous().to(pc.device)
        return rst

    # TODO
    # pc = pc.permute(0,2,1).contiguous()
    rst = cuda_nn.furthest_point_sampling(pc, n_sample)
    return rst

# [b, 3, n] x [m] -> [b, m] x [b, 3, m]
def furthest_sample(pc, n_sample, lazy_sample=True):
    idx = furthest_sample_index(pc, n_sample, lazy_sample)
    return idx, group_nd(pc, idx)


from sklearn.neighbors import NearestNeighbors as nnbrs

def ball_search_np(pc, kpt, knn, search_radius, subsample_ratio=4):
    if subsample_ratio > 1:
        _, pc_sub = uniform_resample_np(pc, pc.shape[0]//subsample_ratio)
    else:
        pc_sub = pc
    # knn-ball search
    nn = min(10000, pc_sub.shape[0])
    nbrs = nnbrs(n_neighbors=nn, algorithm='ball_tree').fit(pc_sub)
    dists, indices = nbrs.kneighbors(pc[kpt])
    true_indices = []
    maxcount = 0
    for i in range(len(dists)):
        if dists[i].max() > search_radius:
            lidx = np.where(dists[i] > search_radius)[0][0]
            # print(f'{lidx} vs {knn}')
            if lidx >= knn:
                true_indices.append(np.random.choice(indices[i][:lidx],knn))
            elif lidx <= 1:
                # ???????
                choice = np.random.choice(range(1), knn - lidx)
                true_indices.append(np.append(indices[i][:lidx], indices[i][choice]))
            else:
                choice = np.random.choice(range(lidx - 1), knn - lidx)
                true_indices.append(np.append(indices[i][:lidx], indices[i][choice]))
        else:
            true_indices.append(np.random.choice(indices[i],knn))
            maxcount += 1

    print("inclusion ratio: ", 1 - float(maxcount)/float(len(dists)))
    return np.array(true_indices, dtype=np.int32), pc_sub

from scipy.spatial import KDTree

def radius_ball_search_np(pc, kpt, search_radius, maxpoints):

    '''
        pc: Nx3
        kpt: Kx3

        return: KxN0x3 patches
    '''

    # radius-ball search
    search = KDTree(pc)
    results = search.query_ball_point(kpt, search_radius)

    all_pc = []
    for indices in results:
        patch = pc[indices]
        if len(indices) > maxpoints:
            patch = subsample_pc(patch, maxpoints)
        all_pc.append(patch)

    return all_pc
