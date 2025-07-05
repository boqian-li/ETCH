import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
from pointnet2_modules import knn_point, farthest_point_sample, gather_operation, group_operation, \
    three_nn, three_interpolate

class MLP(nn.Module):
    def __init__(self, dim, in_channel, mlp, use_bn=True, skip_last=True, last_acti=None):
        super(MLP, self).__init__()
        layers = []
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        bn = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
        last_channel = in_channel
        for i, out_channel in enumerate(mlp):
            layers.append(conv(last_channel, out_channel, 1))
            if use_bn and (not skip_last or i != len(mlp) - 1):
                layers.append(bn(out_channel))
            if (not skip_last or i != len(mlp) - 1):
                layers.append(nn.ReLU())
            last_channel = out_channel
        if last_acti is not None:
            if last_acti == 'softmax':
                layers.append(nn.Softmax(dim=1))
            elif last_acti == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                assert 0, f'Unsupported activation type {last_acti}'
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PointTransformerTransitionDown(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel):
        super(PointTransformerTransitionDown, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = MLP(dim=2, in_channel=in_channel + 3, mlp=[out_channel], use_bn=True, skip_last=False)

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
        _, group_idx = knn_point(self.nsample, new_xyz.transpose(-1, -2), xyz.transpose(-1, -2))
        grouped_xyz = group_operation(xyz, group_idx)  # [B, C, S, nsample]
        grouped_xyz -= new_xyz.view(B, C, S, 1)
        if points is not None:
            grouped_points = group_operation(points, group_idx)   # [B, D, S, nsample]
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=1)
        else:
            grouped_points = grouped_xyz

        grouped_points = self.mlp(grouped_points)  # [B, D, S, nsample]

        new_points = torch.max(grouped_points, -1)[0]  # [B, D', S]

        return new_xyz, new_points


class PointTransformerTransitionUp(nn.Module):
    def __init__(self, low_channel, high_channel):
        super(PointTransformerTransitionUp, self).__init__()
        self.mlp = MLP(dim=1, in_channel=low_channel, mlp=[high_channel], use_bn=True, skip_last=False)

    def forward(self, xyz_low, xyz_high, points_low, points_high):
        """
        Input:
            xyz_high: input points position data, [B, C, N]
            xyz_low: sampled input points position data, [B, C, S]
            points_high: input points data, [B, high_channel, N]
            points_low: input points data, [B, low_channel, S]
        Return:
            new_points: upsampled points data, [B, high_channel, N]
        """
        xyz_high = xyz_high.permute(0, 2, 1)
        xyz_low = xyz_low.permute(0, 2, 1)

        B, N, C = xyz_high.shape
        _, S, _ = xyz_low.shape

        points_low = self.mlp(points_low)
        if S == 1:
            interpolated_points = points_low.repeat(1, 1, N)
        else:
            dist, idx = three_nn(xyz_high, xyz_low)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = three_interpolate(points_low, idx, weight)  # [B, C, N]

        new_points = interpolated_points + points_high
        return new_points


class PointTransformerLayer(nn.Module):
    def __init__( self, dim, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=16):
        super(PointTransformerLayer, self).__init__()
        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Conv1d(dim, dim * 3, 1, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_mlp_hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(pos_mlp_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_mlp_hidden_mult, 1),
            nn.ReLU(),
            nn.Conv2d(dim * attn_mlp_hidden_mult, dim, 1),
        )

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_points: [B, D', N]
        """
        B, C, N = xyz.shape

        _, group_idx = knn_point(self.num_neighbors, xyz.transpose(-1, -2), xyz.transpose(-1, -2))
        grouped_xyz = group_operation(xyz, group_idx)  # [B, C, N, nsample]
        grouped_xyz = xyz.view(B, C, N, 1) - grouped_xyz  # [B, C, N, nsample]
        rel_pos_emb = self.pos_mlp(grouped_xyz)  # [B, D', N, nsample]
        # get queries, keys, values
        q, k, v = self.to_qkv(points).chunk(3, dim=-2)  # [B, 3 * D', N] -> 3 * [B, D', N]
        qk_rel = q.view(B, -1, N, 1) - group_operation(k, group_idx)  # [B, D', N, nsample]
        v = group_operation(v, group_idx)  # [B, D', N, nsample]

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)  # [B, D', N, nsample]
        attn = sim.softmax(dim=-1)
        agg = torch.sum(attn * v, dim=-1)  # [B, D', N]
        return agg


class PointTransformerResBlock(nn.Module):
    def __init__( self, dim, div=4, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=16):
        super(PointTransformerResBlock, self).__init__()
        mid_dim = dim // div
        self.transformer_layer = PointTransformerLayer(mid_dim, pos_mlp_hidden_dim,
                                                       attn_mlp_hidden_mult, num_neighbors)
        self.before_mlp = nn.Conv1d(dim, mid_dim, 1)
        self.after_mlp = nn.Conv1d(mid_dim, dim, 1)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_points: [B, D', N]
        """
        input_points = points
        points = self.before_mlp(points)
        points = self.transformer_layer(xyz, points)
        points = self.after_mlp(points)
        return input_points + points


class PointTransformerDownBlock(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, num_attn, div=4, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4):
        super(PointTransformerDownBlock, self).__init__()
        self.down = PointTransformerTransitionDown(npoint, nsample, in_channel, out_channel)
        attn = []
        for i in range(num_attn):
            attn.append(PointTransformerResBlock(out_channel, div, pos_mlp_hidden_dim, attn_mlp_hidden_mult, nsample))
        self.attn = nn.ModuleList(attn)

    def forward(self, xyz, points):
        xyz, points = self.down(xyz, points)
        for layer in self.attn:
            points = layer(xyz, points)

        return xyz, points


class PointTransformerUpBlock(nn.Module):
    def __init__(self, nsample, low_channel, high_channel, num_attn, div=4, pos_mlp_hidden_dim=64,
                 attn_mlp_hidden_mult=4):
        super(PointTransformerUpBlock, self).__init__()
        self.up = PointTransformerTransitionUp(low_channel, high_channel)
        attn = []
        for i in range(num_attn):
            attn.append(PointTransformerResBlock(high_channel, div, pos_mlp_hidden_dim, attn_mlp_hidden_mult, nsample))
        self.attn = nn.ModuleList(attn)

    def forward(self, xyz_low, xyz_high, points_low, points_high):
        points = self.up(xyz_low, xyz_high, points_low, points_high)
        for layer in self.attn:
            points = layer(xyz_high, points)

        return points
