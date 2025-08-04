import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)


def bp():
    import pdb;pdb.set_trace()

# bp()
CUDA = torch.cuda.is_available()
if CUDA:
    import pointnet2_utils as futils

def knn_point(k, pos2, pos1):
    '''
    Input:
        k: int32, number of k in k-nn search
        pos1: (batch_size, ndataset, c) float32 array, input points
        pos2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    if CUDA:
        val, idx = futils.knn(k, pos2, pos1)
        return val, idx.long()

    B, N, C = pos1.shape
    M = pos2.shape[1]
    pos1 = pos1.view(B, 1, N, -1).repeat(1, M, 1, 1)
    pos2 = pos2.view(B, M, 1, -1).repeat(1, 1, N, 1)
    dist = torch.sum(-(pos1 - pos2) ** 2, -1)
    val, idx = dist.topk(k=k, dim=-1)
    return torch.sqrt(-val), idx


def three_nn(xyz1, xyz2):
    if CUDA:
        dists, idx = futils.three_nn(xyz1, xyz2)
        return dists, idx.long()

    dists = square_distance(xyz1, xyz2)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
    return dists, idx


def three_interpolate(points, idx, weight):  # points: [B, C, M], idx: [B, N, 3], returns [B, C, N]
    if CUDA:
        return futils.three_interpolate(points, idx.int(), weight)

    B, N = idx.shape[:2]
    points = points.permute(0, 2, 1)  # [B, M, C] --> [B, N, 3, C]
    interpolated_points = torch.sum(index_points(points, idx) * weight.view(B, N, 3, 1), dim=2)
    return interpolated_points.permute(0, 2, 1)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S1, S2, ..Sk]
    Return:
        new_points:, indexed points data, [B, S, C] or [B, S1, S2, ..Sk, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def gather_operation(feature, idx):  # [B, C, N], [B, npoint] -> [B, C, npoint]
    if CUDA:
        return futils.gather_operation(feature, idx)
    return index_points(feature.transpose(-1, -2), idx).transpose(-1, -2)


def group_operation(feature, idx):  # [B, C, N], idx [B, npoint, nsample] --> [B, C, npoint, nsample]
    if CUDA:
        return futils.grouping_operation(feature, idx)
    return index_points(feature.transpose(-1, -2), idx).permute(0, 3, 1, 2)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    if CUDA:
        idx = futils.furthest_point_sample(xyz, npoint).long()
        return idx

    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    if CUDA:
        return futils.ball_query(radius, nsample, xyz, new_xyz).long()

    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask_first = group_first == N
    group_first[mask_first] = 0
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, knn=False):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.out_channel = 0
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.out_channel += last_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
        self.knn = knn

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        B, C, N = xyz.shape
        S = self.npoint
        fps_idx = farthest_point_sample(xyz.permute(0, 2, 1), S).int()
        new_xyz = gather_operation(xyz, fps_idx)  # [B, C, S]
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            if self.knn:
                _, group_idx = knn_point(K, new_xyz.transpose(-1, -2), xyz.transpose(-1, -2))
            else:
                group_idx = query_ball_point(radius, K, xyz.transpose(-1, -2), new_xyz.transpose(-1, -2))  # [B, S, nsample]
            grouped_xyz = group_operation(xyz, group_idx)  # [B, C, S, nsample]
            grouped_xyz -= new_xyz.view(B, C, S, 1)
            if points is not None:
                grouped_points = group_operation(points, group_idx)   # [B, D, S, nsample]
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=1)
            else:
                grouped_points = grouped_xyz

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))  # [B, D, S, nsample]
            new_points = torch.max(grouped_points, -1)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.out_channel = last_channel
        self.group_all = group_all
        self.knn = knn

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            assert 0, 'Not Implemented'

        new_points = new_points.permute(0, 3, 2, 1)  # [B, 1, N, 3 + D] --> [B, 3 + D, N, 1]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.out_channel = last_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dist, idx = three_nn(xyz1, xyz2)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = three_interpolate(points2, idx, weight)  # [B, C, N]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-2)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
