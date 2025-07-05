# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch

"""
This file defines distances between meshes and pointclouds.
The functions make use of the definition of a distance between a point and
an edge segment or the distance of a point and a triangle (face).

The exact mathematical formulations and implementations of these
distances can be found in `csrc/utils/geometry_utils.cuh`.
"""

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3




def my_point_mesh_face_distance(
    meshes: Meshes,
    pcls: Pointclouds,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    计算点云到网格的距离
    Args:
        meshes: Meshes object
        pcls: Pointclouds object
        min_triangle_area: minimum triangle area threshold
    Returns:
        point_to_face: (B, N) 每个点到最近面的距离
        face_to_point: (B, F) 每个面到最近点的距离
    """
    # Get batch information
    B = len(meshes)
    N = pcls.num_points_per_cloud()[0]  # All batches have the same number of points
    F = meshes.num_faces_per_mesh()[0]  # All batches have the same number of faces
    
    # Get point and face data
    points = pcls.points_padded()  # (B, N, 3)
    
    # Get triangle vertices
    verts_padded = meshes.verts_padded()  # (B, V, 3)
    faces_padded = meshes.faces_padded()  # (B, F, 3)
    
    # Build triangle vertex coordinates
    tris = torch.stack([
        torch.gather(verts_padded, 1, faces_padded[..., 0:1].expand(-1, -1, 3)),
        torch.gather(verts_padded, 1, faces_padded[..., 1:2].expand(-1, -1, 3)),
        torch.gather(verts_padded, 1, faces_padded[..., 2:3].expand(-1, -1, 3))
    ], dim=2)  # (B, F, 3, 3)
    
    # Calculate triangle normals
    v1 = tris[:, :, 1] - tris[:, :, 0]  # (B, F, 3)
    v2 = tris[:, :, 2] - tris[:, :, 0]  # (B, F, 3)
    normals = torch.cross(v1, v2, dim=-1)  # (B, F, 3)
    
    # Normalize normals
    normal_lengths = torch.norm(normals, dim=-1, keepdim=True)  # (B, F, 1)
    valid_triangles = normal_lengths > 1e-8  # (B, F, 1)
    normals = normals / (normal_lengths + 1e-8)  # (B, F, 3)
    
    # Calculate distance from point to triangle plane
    points_expanded = points.unsqueeze(2)  # (B, N, 1, 3)
    v = points_expanded - tris[:, None, :, 0]  # (B, N, F, 3)
    dist_to_plane = torch.abs(torch.sum(v * normals.unsqueeze(1), dim=-1))  # (B, N, F)
    
    # Calculate projection points
    proj_points = points_expanded - dist_to_plane.unsqueeze(-1) * normals.unsqueeze(1)  # (B, N, F, 3)
    
    # Calculate barycentric coordinates
    edge1 = tris[:, :, 1] - tris[:, :, 0]  # (B, F, 3)
    edge2 = tris[:, :, 2] - tris[:, :, 0]  # (B, F, 3)
    v2p = proj_points - tris[:, None, :, 0]  # (B, N, F, 3)
    
    d00 = torch.sum(edge1 * edge1, dim=-1)  # (B, F)
    d01 = torch.sum(edge1 * edge2, dim=-1)  # (B, F)
    d11 = torch.sum(edge2 * edge2, dim=-1)  # (B, F)
    d20 = torch.sum(v2p * edge1.unsqueeze(1), dim=-1)  # (B, N, F)
    d21 = torch.sum(v2p * edge2.unsqueeze(1), dim=-1)  # (B, N, F)
    
    denom = d00.unsqueeze(1) * d11.unsqueeze(1) - d01.unsqueeze(1) * d01.unsqueeze(1)  # (B, 1, F)
    v = (d11.unsqueeze(1) * d20 - d01.unsqueeze(1) * d21) / (denom + 1e-8)  # (B, N, F)
    w = (d00.unsqueeze(1) * d21 - d01.unsqueeze(1) * d20) / (denom + 1e-8)  # (B, N, F)
    u = 1.0 - v - w  # (B, N, F)
    
    # Check if point is inside triangle
    in_triangle = (u >= 0) & (v >= 0) & (w >= 0) & (u <= 1) & (v <= 1) & (w <= 1)  # (B, N, F)
    
    # Calculate distance to edges
    edge_distances = torch.full((B, N, F), float('inf'), device=points.device)
    
    # Calculate distance to three edges
    edges = [
        (tris[:, :, 0], tris[:, :, 1]),
        (tris[:, :, 1], tris[:, :, 2]),
        (tris[:, :, 2], tris[:, :, 0])
    ]
    
    for start, end in edges:
        edge_vec = end - start  # (B, F, 3)
        point_vec = points_expanded - start.unsqueeze(1)  # (B, N, F, 3)
        
        edge_len_sq = torch.sum(edge_vec * edge_vec, dim=-1) + 1e-8  # (B, F)
        t = torch.sum(point_vec * edge_vec.unsqueeze(1), dim=-1) / edge_len_sq.unsqueeze(1)  # (B, N, F)
        t = torch.clamp(t, 0, 1)
        
        proj = start.unsqueeze(1) + t.unsqueeze(-1) * edge_vec.unsqueeze(1)  # (B, N, F, 3)
        edge_dist = torch.norm(points_expanded - proj, dim=-1)  # (B, N, F)
        edge_distances = torch.minimum(edge_distances, edge_dist)
    
    # Combine plane distance and edge distance
    distances = torch.where(in_triangle, dist_to_plane, edge_distances)  # (B, N, F)
    valid_triangles = valid_triangles.squeeze(-1)  # (B, F)
    distances = torch.where(valid_triangles.unsqueeze(1), distances, 
                          torch.full_like(distances, float('inf')))  # (B, N, F)
    
    # Calculate final distance
    point_to_face, _ = torch.min(distances, dim=2)  # (B, N)
    face_to_point, _ = torch.min(distances, dim=1)  # (B, F)
    
    return point_to_face, face_to_point  # (B, N), (B, F)

















# # PointFaceDistance
# class _PointFaceDistance(Function):
#     """
#     Torch autograd Function wrapper PointFaceDistance Cuda implementation
#     """

#     @staticmethod
#     def forward(
#         ctx,
#         points,
#         points_first_idx,
#         tris,
#         tris_first_idx,
#         max_points,
#         min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
#     ):
#         """
#         Args:
#             ctx: Context object used to calculate gradients.
#             points: FloatTensor of shape `(P, 3)`
#             points_first_idx: LongTensor of shape `(N,)` indicating the first point
#                 index in each example in the batch
#             tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
#                 triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
#             tris_first_idx: LongTensor of shape `(N,)` indicating the first face
#                 index in each example in the batch
#             max_points: Scalar equal to maximum number of points in the batch
#             min_triangle_area: (float, defaulted) Triangles of area less than this
#                 will be treated as points/lines.
#         Returns:
#             dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
#                 euclidean distance of `p`-th point to the closest triangular face
#                 in the corresponding example in the batch
#             idxs: LongTensor of shape `(P,)` indicating the closest triangular face
#                 in the corresponding example in the batch.

#             `dists[p]` is
#             `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
#             where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
#             face `(v0, v1, v2)`

#         """
#         dists, idxs = _C.point_face_dist_forward(
#             points,
#             points_first_idx,
#             tris,
#             tris_first_idx,
#             max_points,
#             min_triangle_area,
#         )
#         ctx.save_for_backward(points, tris, idxs)
#         ctx.min_triangle_area = min_triangle_area
#         return dists

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_dists):
#         grad_dists = grad_dists.contiguous()
#         points, tris, idxs = ctx.saved_tensors
#         min_triangle_area = ctx.min_triangle_area
#         grad_points, grad_tris = _C.point_face_dist_backward(
#             points, tris, idxs, grad_dists, min_triangle_area
#         )
#         return grad_points, None, grad_tris, None, None, None


# point_face_distance = _PointFaceDistance.apply


# # FacePointDistance
# class _FacePointDistance(Function):
#     """
#     Torch autograd Function wrapper FacePointDistance Cuda implementation
#     """

#     @staticmethod
#     def forward(
#         ctx,
#         points,
#         points_first_idx,
#         tris,
#         tris_first_idx,
#         max_tris,
#         min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
#     ):
#         """
#         Args:
#             ctx: Context object used to calculate gradients.
#             points: FloatTensor of shape `(P, 3)`
#             points_first_idx: LongTensor of shape `(N,)` indicating the first point
#                 index in each example in the batch
#             tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
#                 triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
#             tris_first_idx: LongTensor of shape `(N,)` indicating the first face
#                 index in each example in the batch
#             max_tris: Scalar equal to maximum number of faces in the batch
#             min_triangle_area: (float, defaulted) Triangles of area less than this
#                 will be treated as points/lines.
#         Returns:
#             dists: FloatTensor of shape `(T,)`, where `dists[t]` is the squared
#                 euclidean distance of `t`-th triangular face to the closest point in the
#                 corresponding example in the batch
#             idxs: LongTensor of shape `(T,)` indicating the closest point in the
#                 corresponding example in the batch.

#             `dists[t] = d(points[idxs[t]], tris[t, 0], tris[t, 1], tris[t, 2])`,
#             where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
#             face `(v0, v1, v2)`.
#         """
#         dists, idxs = _C.face_point_dist_forward(
#             points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
#         )
#         ctx.save_for_backward(points, tris, idxs)
#         ctx.min_triangle_area = min_triangle_area
#         return dists

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_dists):
#         grad_dists = grad_dists.contiguous()
#         points, tris, idxs = ctx.saved_tensors
#         min_triangle_area = ctx.min_triangle_area
#         grad_points, grad_tris = _C.face_point_dist_backward(
#             points, tris, idxs, grad_dists, min_triangle_area
#         )
#         return grad_points, None, grad_tris, None, None, None


# face_point_distance = _FacePointDistance.apply

# def my_point_mesh_face_distance(
#     meshes: Meshes,
#     pcls: Pointclouds,
#     min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
# ):
#     if len(meshes) != len(pcls):
#         raise ValueError("meshes and pointclouds must be equal sized batches")
#     N = len(meshes)

#     # 获取packed数据
#     points = pcls.points_packed()  # (sum(P_i), 3)
#     points_first_idx = pcls.cloud_to_packed_first_idx()
#     point_to_cloud_idx = pcls.packed_to_cloud_idx()

#     verts_packed = meshes.verts_packed()
#     faces_packed = meshes.faces_packed()
#     tris = verts_packed[faces_packed]  # (sum(T_i), 3, 3)
#     tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
#     tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()

#     # 计算点到面的距离
#     P = points.shape[0]  # 总点数
#     T = tris.shape[0]    # 总面数
    
#     # 计算三角形的法向量
#     v1 = tris[:, 1] - tris[:, 0]  # (T, 3)
#     v2 = tris[:, 2] - tris[:, 0]  # (T, 3)
#     normals = torch.cross(v1, v2, dim=1)  # (T, 3)
    
#     # 单位化法向量
#     normal_lengths = torch.norm(normals, dim=1, keepdim=True)  # (T, 1)
#     valid_triangles = normal_lengths > 1e-8    # (T, 1)
#     normals = normals / (normal_lengths + 1e-8)  # (T, 3)
    
#     # 计算点到三角形平面的距离
#     points_expanded = points.unsqueeze(1)  # (P, 1, 3)
#     v = points_expanded - tris[:, 0].unsqueeze(0)  # (P, T, 3)
#     dist_to_plane = torch.abs(torch.sum(v * normals.unsqueeze(0), dim=2))  # (P, T)
    
#     # 计算投影点
#     proj_points = points_expanded - dist_to_plane.unsqueeze(2) * normals.unsqueeze(0)
    
#     # 计算重心坐标
#     edge1 = tris[:, 1] - tris[:, 0]  # (T, 3)
#     edge2 = tris[:, 2] - tris[:, 0]  # (T, 3)
#     v2p = proj_points - tris[:, 0].unsqueeze(0)  # (P, T, 3)
    
#     d00 = torch.sum(edge1 * edge1, dim=1)  # (T,)
#     d01 = torch.sum(edge1 * edge2, dim=1)  # (T,)
#     d11 = torch.sum(edge2 * edge2, dim=1)  # (T,)
#     d20 = torch.sum(v2p * edge1.unsqueeze(0), dim=2)  # (P, T)
#     d21 = torch.sum(v2p * edge2.unsqueeze(0), dim=2)  # (P, T)
    
#     denom = d00 * d11 - d01 * d01  # (T,)
#     v = (d11 * d20 - d01 * d21) / (denom + 1e-8)  # (P, T)
#     w = (d00 * d21 - d01 * d20) / (denom + 1e-8)  # (P, T)
#     u = 1.0 - v - w
    
#     # 判断是否在三角形内
#     in_triangle = (u >= 0) & (v >= 0) & (w >= 0) & (u <= 1) & (v <= 1) & (w <= 1)  # (P, T)
    
#     # 计算到边的距离
#     edges = [
#         (tris[:, 0], tris[:, 1]),
#         (tris[:, 1], tris[:, 2]),
#         (tris[:, 2], tris[:, 0])
#     ]
    
#     edge_distances = torch.full((P, T), float('inf'), device=points.device)
    
#     for start, end in edges:
#         edge_vec = end - start  # (T, 3)
#         point_vec = points_expanded - start.unsqueeze(0)  # (P, T, 3)
        
#         edge_len_sq = torch.sum(edge_vec * edge_vec, dim=1) + 1e-8  # (T,)
#         t = torch.sum(point_vec * edge_vec.unsqueeze(0), dim=2) / edge_len_sq  # (P, T)
#         t = torch.clamp(t, 0, 1)
        
#         proj = start.unsqueeze(0) + t.unsqueeze(2) * edge_vec.unsqueeze(0)
#         edge_dist = torch.norm(points_expanded - proj, dim=2)
#         edge_distances = torch.minimum(edge_distances, edge_dist)
    
#     # 组合平面距离和边距离
#     distances = torch.where(in_triangle, dist_to_plane, edge_distances)  # (P, T)
#     valid_triangles = valid_triangles.squeeze(1)  # (T,)
#     distances = torch.where(valid_triangles.unsqueeze(0), distances, 
#                           torch.full((P, T), float('inf'), device=points.device))
    
#     # 计算每个点到最近面的距离和每个面到最近点的距离
#     point_to_face, _ = torch.min(distances, dim=1)  # (P,)
#     face_to_point, _ = torch.min(distances, dim=0)  # (T,)

#     # 应用权重
#     weights_p = 1.0 / pcls.num_points_per_cloud().float().gather(0, point_to_cloud_idx)
#     weights_t = 1.0 / meshes.num_faces_per_mesh().float().gather(0, tri_to_mesh_idx)
    
#     point_to_face = point_to_face * weights_p
#     face_to_point = face_to_point * weights_t

#     # 重塑结果为每个batch
#     point_to_face_per_cloud = torch.zeros_like(point_to_cloud_idx, dtype=torch.float32)
#     face_to_point_per_mesh = torch.zeros_like(tri_to_mesh_idx, dtype=torch.float32)
    
#     point_to_face_per_cloud.scatter_(0, point_to_cloud_idx, point_to_face)
#     face_to_point_per_mesh.scatter_(0, tri_to_mesh_idx, face_to_point)

#     return point_to_face_per_cloud, face_to_point_per_mesh


# def point_to_triangle_distance(points, triangles, points_first_idx, tris_first_idx):
#     """
#     计算点到三角形的距离，考虑packed数据结构
#     points: (sum(P_i), 3) packed points
#     triangles: (sum(T_i), 3, 3) packed triangles
#     points_first_idx: (B,) 每个batch中第一个点的索引
#     tris_first_idx: (B,) 每个batch中第一个三角形的索引
#     """
#     # 获取batch size
#     B = len(points_first_idx)
    
#     # 准备存储结果
#     all_point_to_face = []
#     all_face_to_point = []
    
#     for b in range(B):
#         # 获取当前batch的点和三角形
#         p_start = points_first_idx[b]
#         p_end = points_first_idx[b + 1] if b < B - 1 else points.shape[0]
#         t_start = tris_first_idx[b]
#         t_end = tris_first_idx[b + 1] if b < B - 1 else triangles.shape[0]
        
#         batch_points = points[p_start:p_end]      # (P_i, 3)
#         batch_tris = triangles[t_start:t_end]     # (T_i, 3, 3)
        
#         P = batch_points.shape[0]  # 当前batch中的点数
#         T = batch_tris.shape[0]    # 当前batch中的三角形数
        
#         # 计算三角形的法向量
#         v1 = batch_tris[:, 1] - batch_tris[:, 0]  # (T_i, 3)
#         v2 = batch_tris[:, 2] - batch_tris[:, 0]  # (T_i, 3)
#         normals = torch.cross(v1, v2, dim=1)      # (T_i, 3)
        
#         # 单位化法向量
#         normal_lengths = torch.norm(normals, dim=1, keepdim=True)  # (T_i, 1)
#         valid_triangles = normal_lengths > 1e-8    # (T_i, 1)
#         normals = normals / (normal_lengths + 1e-8)  # (T_i, 3)
        
#         # 计算点到三角形平面的距离
#         points_expanded = batch_points.unsqueeze(1)  # (P_i, 1, 3)
#         v = points_expanded - batch_tris[:, 0].unsqueeze(0)  # (P_i, T_i, 3)
#         dist_to_plane = torch.abs(torch.sum(v * normals.unsqueeze(0), dim=2))  # (P_i, T_i)
        
#         # 计算投影点
#         proj_points = points_expanded - dist_to_plane.unsqueeze(2) * normals.unsqueeze(0)
        
#         # 计算重心坐标
#         edge1 = batch_tris[:, 1] - batch_tris[:, 0]  # (T_i, 3)
#         edge2 = batch_tris[:, 2] - batch_tris[:, 0]  # (T_i, 3)
#         v2p = proj_points - batch_tris[:, 0].unsqueeze(0)  # (P_i, T_i, 3)
        
#         d00 = torch.sum(edge1 * edge1, dim=1)  # (T_i,)
#         d01 = torch.sum(edge1 * edge2, dim=1)  # (T_i,)
#         d11 = torch.sum(edge2 * edge2, dim=1)  # (T_i,)
#         d20 = torch.sum(v2p * edge1.unsqueeze(0), dim=2)  # (P_i, T_i)
#         d21 = torch.sum(v2p * edge2.unsqueeze(0), dim=2)  # (P_i, T_i)
        
#         denom = d00 * d11 - d01 * d01  # (T_i,)
#         v = (d11 * d20 - d01 * d21) / (denom + 1e-8)  # (P_i, T_i)
#         w = (d00 * d21 - d01 * d20) / (denom + 1e-8)  # (P_i, T_i)
#         u = 1.0 - v - w
        
#         # 判断是否在三角形内
#         in_triangle = (u >= 0) & (v >= 0) & (w >= 0) & (u <= 1) & (v <= 1) & (w <= 1)  # (P_i, T_i)
        
#         # 计算到边的距离（无条件语句）
#         edges = [
#             (batch_tris[:, 0], batch_tris[:, 1]),
#             (batch_tris[:, 1], batch_tris[:, 2]),
#             (batch_tris[:, 2], batch_tris[:, 0])
#         ]
        
#         edge_distances = torch.full((P, T), float('inf'), device=points.device)  # (P_i, T_i)
        
#         for start, end in edges:
#             edge_vec = end - start  # (T_i, 3)
#             point_vec = points_expanded - start.unsqueeze(0)  # (P_i, T_i, 3)
            
#             edge_len_sq = torch.sum(edge_vec * edge_vec, dim=1) + 1e-8  # (T_i,)
#             t = torch.sum(point_vec * edge_vec.unsqueeze(0), dim=2) / edge_len_sq  # (P_i, T_i)
#             t = torch.clamp(t, 0, 1)
            
#             proj = start.unsqueeze(0) + t.unsqueeze(2) * edge_vec.unsqueeze(0)
#             edge_dist = torch.norm(points_expanded - proj, dim=2)
#             edge_distances = torch.minimum(edge_distances, edge_dist)
        
#         # 组合平面距离和边距离
#         distances = torch.where(in_triangle, dist_to_plane, edge_distances)  # (P_i, T_i)
#         valid_triangles = valid_triangles.squeeze(1)  # (T_i,)
#         distances = torch.where(valid_triangles.unsqueeze(0), distances, 
#                               torch.full((P, T), float('inf'), device=points.device))  # (P_i, T_i)
        
#         # 计算每个点到最近面的距离
#         point_to_face, _ = torch.min(distances, dim=1)  # (P_i,)
#         # 计算每个面到最近点的距离
#         face_to_point, _ = torch.min(distances, dim=0)  # (T_i,)
        
#         all_point_to_face.append(point_to_face)
#         all_face_to_point.append(face_to_point)
    
#     return all_point_to_face, all_face_to_point

# def my_point_mesh_face_distance(
#     meshes: Meshes,
#     pcls: Pointclouds,
#     min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
# ):
#     if len(meshes) != len(pcls):
#         raise ValueError("meshes and pointclouds must be equal sized batches")
#     N = len(meshes)

#     # 获取packed数据
#     points = pcls.points_packed()  # (sum(P_i), 3)
#     points_first_idx = pcls.cloud_to_packed_first_idx()
#     point_to_cloud_idx = pcls.packed_to_cloud_idx()

#     verts_packed = meshes.verts_packed()
#     faces_packed = meshes.faces_packed()
#     tris = verts_packed[faces_packed]  # (sum(T_i), 3, 3)
#     tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
#     tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()

#     # 计算距离
#     point_to_face_list, face_to_point_list = point_to_triangle_distance(
#         points, tris, points_first_idx, tris_first_idx
#     )

#     # 应用权重并重塑结果
#     point_to_face = torch.cat(point_to_face_list)
#     face_to_point = torch.cat(face_to_point_list)

#     # 应用权重
#     num_points_per_cloud = pcls.num_points_per_cloud()
#     weights_p = 1.0 / num_points_per_cloud.float().gather(0, point_to_cloud_idx)
#     point_to_face = point_to_face * weights_p

#     num_tris_per_mesh = meshes.num_faces_per_mesh()
#     weights_t = 1.0 / num_tris_per_mesh.float().gather(0, tri_to_mesh_idx)
#     face_to_point = face_to_point * weights_t

#     # 重塑为每个batch的形状
#     # 使用 cumsum 来获取分割点
#     points_per_cloud = pcls.num_points_per_cloud()
#     faces_per_mesh = meshes.num_faces_per_mesh()
    
#     # 创建结果张量
#     max_points = points_per_cloud.max()
#     max_faces = faces_per_mesh.max()
    
#     point_to_face_batched = torch.zeros((N, max_points), device=points.device)
#     face_to_point_batched = torch.zeros((N, max_faces), device=points.device)
    
#     # 填充结果
#     start_idx = 0
#     for i in range(N):
#         num_points = points_per_cloud[i]
#         point_to_face_batched[i, :num_points] = point_to_face_list[i]
        
#         num_faces = faces_per_mesh[i]
#         face_to_point_batched[i, :num_faces] = face_to_point_list[i]

#     return point_to_face_batched, face_to_point_batched

# def point_to_triangle_distance(points, triangles):
#     """
#     计算点到三角形的距离
#     points: (P, 3)
#     triangles: (T, 3, 3)
#     返回: (P, T) 每个点到每个三角形的距离
#     """
#     # 计算三角形的法向量
#     v1 = triangles[:, 1] - triangles[:, 0]  # (T, 3)
#     v2 = triangles[:, 2] - triangles[:, 0]  # (T, 3)
#     normals = torch.cross(v1, v2, dim=1)  # (T, 3)
    
#     # 单位化法向量
#     normal_lengths = torch.norm(normals, dim=1, keepdim=True)
#     valid_triangles = normal_lengths > 1e-8
#     normals = normals / (normal_lengths + 1e-8)  # (T, 3)

#     # 计算点到三角形平面的距离
#     points_expanded = points.unsqueeze(1)  # (P, 1, 3)
#     v = points_expanded - triangles[:, 0].unsqueeze(0)  # (P, T, 3)
#     dist_to_plane = torch.abs(torch.sum(v * normals.unsqueeze(0), dim=2))  # (P, T)

#     # 计算投影点
#     proj_points = points_expanded - dist_to_plane.unsqueeze(2) * normals.unsqueeze(0)  # (P, T, 3)

#     # 使用重心坐标判断投影点是否在三角形内
#     edge1 = triangles[:, 1] - triangles[:, 0]  # (T, 3)
#     edge2 = triangles[:, 2] - triangles[:, 0]  # (T, 3)
#     v2p = proj_points - triangles[:, 0].unsqueeze(0)  # (P, T, 3)

#     # 计算重心坐标
#     d00 = torch.sum(edge1 * edge1, dim=1)  # (T,)
#     d01 = torch.sum(edge1 * edge2, dim=1)  # (T,)
#     d11 = torch.sum(edge2 * edge2, dim=1)  # (T,)
#     d20 = torch.sum(v2p * edge1.unsqueeze(0), dim=2)  # (P, T)
#     d21 = torch.sum(v2p * edge2.unsqueeze(0), dim=2)  # (P, T)

#     denom = d00 * d11 - d01 * d01  # (T,)
#     v = (d11 * d20 - d01 * d21) / (denom + 1e-8)  # (P, T)
#     w = (d00 * d21 - d01 * d20) / (denom + 1e-8)  # (P, T)
#     u = 1.0 - v - w

#     # 判断是否在三角形内
#     in_triangle = (u >= 0) & (v >= 0) & (w >= 0) & (u <= 1) & (v <= 1) & (w <= 1)  # (P, T)

#     # 如果不在三角形内，计算到边的最短距离
#     edge_distances = torch.zeros_like(dist_to_plane)  # (P, T)
#     outside_triangle = ~in_triangle

#     if outside_triangle.any():
#         edges = [
#             (triangles[:, 0], triangles[:, 1]),
#             (triangles[:, 1], triangles[:, 2]),
#             (triangles[:, 2], triangles[:, 0])
#         ]
        
#         for start, end in edges:
#             edge_vec = end - start  # (T, 3)
#             point_vec = points_expanded - start.unsqueeze(0)  # (P, T, 3)
            
#             t = torch.sum(point_vec * edge_vec.unsqueeze(0), dim=2) / (
#                 torch.sum(edge_vec * edge_vec, dim=1) + 1e-8
#             )  # (P, T)
#             t = torch.clamp(t, 0, 1)
            
#             proj = start.unsqueeze(0) + t.unsqueeze(2) * edge_vec.unsqueeze(0)  # (P, T, 3)
#             edge_dist = torch.norm(points_expanded - proj, dim=2)  # (P, T)
            
#             edge_distances = torch.min(edge_distances, edge_dist)

#     # 组合平面距离和边距离
#     distances = torch.where(in_triangle, dist_to_plane, edge_distances)
#     distances = torch.where(valid_triangles.unsqueeze(0), distances, torch.full_like(distances, float('inf')))
    
#     return distances

# def my_point_mesh_face_distance(
#     meshes: Meshes,
#     pcls: Pointclouds,
#     min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
# ):
#     if len(meshes) != len(pcls):
#         raise ValueError("meshes and pointclouds must be equal sized batches")
#     N = len(meshes)

#     # 获取点云数据
#     points = pcls.points_packed()  # (P, 3)
#     points_first_idx = pcls.cloud_to_packed_first_idx()
#     point_to_cloud_idx = pcls.packed_to_cloud_idx()

#     # 获取网格数据
#     verts_packed = meshes.verts_packed()
#     faces_packed = meshes.faces_packed()
#     tris = verts_packed[faces_packed]  # (T, 3, 3)
#     tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()

#     # 计算点到面的距离
#     all_distances = point_to_triangle_distance(points, tris)  # (P, T)

#     # 对每个点找最近的面
#     point_to_face, _ = torch.min(all_distances, dim=1)  # (P,)

#     # 对每个面找最近的点
#     face_to_point, _ = torch.min(all_distances, dim=0)  # (T,)

#     # 应用权重
#     num_points_per_cloud = pcls.num_points_per_cloud()
#     weights_p = 1.0 / num_points_per_cloud.float().gather(0, point_to_cloud_idx)
#     point_to_face = point_to_face * weights_p

#     num_tris_per_mesh = meshes.num_faces_per_mesh()
#     weights_t = 1.0 / num_tris_per_mesh.float().gather(0, tri_to_mesh_idx)
#     face_to_point = face_to_point * weights_t

#     # 重塑输出
#     points_per_cloud = pcls.num_points_per_cloud()
#     faces_per_mesh = meshes.num_faces_per_mesh()
    
#     point_to_face = torch.split(point_to_face, points_per_cloud.tolist())
#     face_to_point = torch.split(face_to_point, faces_per_mesh.tolist())
    
#     point_to_face = torch.nn.utils.rnn.pad_sequence(point_to_face, batch_first=True)
#     face_to_point = torch.nn.utils.rnn.pad_sequence(face_to_point, batch_first=True)

#     return point_to_face, face_to_point


# def my_point_mesh_face_distance(
#     meshes: Meshes,
#     pcls: Pointclouds,
#     min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
# ):
#     """
#     Computes the distance between a pointcloud and a mesh within a batch.
#     Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
#     sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

#     `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
#         to the closest triangular face in mesh and averages across all points in pcl
#     `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
#         mesh to the closest point in pcl and averages across all faces in mesh.

#     The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
#     and then averaged across the batch.

#     Args:
#         meshes: A Meshes data structure containing N meshes
#         pcls: A Pointclouds data structure containing N pointclouds
#         min_triangle_area: (float, defaulted) Triangles of area less than this
#             will be treated as points/lines.

#     Returns:
#         loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
#             between all `(mesh, pcl)` in a batch averaged across the batch.
#     """

#     if len(meshes) != len(pcls):
#         raise ValueError("meshes and pointclouds must be equal sized batches")
#     N = len(meshes)

#     # packed representation for pointclouds
#     points = pcls.points_packed()  # (P, 3)
#     points_first_idx = pcls.cloud_to_packed_first_idx()
#     max_points = pcls.num_points_per_cloud().max().item()

#     # packed representation for faces
#     verts_packed = meshes.verts_packed()
#     faces_packed = meshes.faces_packed()
#     tris = verts_packed[faces_packed]  # (T, 3, 3)
#     tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
#     max_tris = meshes.num_faces_per_mesh().max().item()

#     # point to face distance: shape (P,)
#     # point_to_face = point_face_distance(
#     #     points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
#     # )
#     # point_to_face, idxs = _C.point_face_dist_forward(
#     #         points,
#     #         points_first_idx,
#     #         tris,
#     #         tris_first_idx,
#     #         max_points,
#     #         min_triangle_area,
#     #     )

#     # weight each example by the inverse of number of points in the example
#     point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
#     num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
#     weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
#     # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
#     weights_p = 1.0 / weights_p.float()
#     point_to_face = point_to_face * weights_p
#     # point_dist = point_to_face.sum() / N

#     # face to point distance: shape (T,)
#     # face_to_point = face_point_distance(
#     #     points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
#     # )
#     face_to_point, idxs = _C.face_point_dist_forward(
#             points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
#         )

#     # weight each example by the inverse of number of faces in the example
#     tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
#     num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
#     weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
#     weights_t = 1.0 / weights_t.float()
#     face_to_point = face_to_point * weights_t
#     # face_dist = face_to_point.sum() / N

#     return point_to_face.reshape(N, -1), face_to_point.reshape(N, -1) # shape (N, num_points), (N, num_faces)


################################################################
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# # pyre-unsafe

# from pytorch3d import _C
# from pytorch3d.structures import Meshes, Pointclouds
# from torch.autograd import Function
# from torch.autograd.function import once_differentiable
# import torch
# import functorch

# """
# This file defines distances between meshes and pointclouds.
# The functions make use of the definition of a distance between a point and
# an edge segment or the distance of a point and a triangle (face).

# The exact mathematical formulations and implementations of these
# distances can be found in `csrc/utils/geometry_utils.cuh`.
# """

# _DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


# # PointFaceDistance
# class _PointFaceDistance(Function):
#     """
#     Torch autograd Function wrapper PointFaceDistance Cuda implementation
#     """
#     # generate_vmap_rule = True

#     @staticmethod
#     def forward(
#         points,
#         points_first_idx,
#         tris,
#         tris_first_idx,
#         max_points,
#         min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
#     ):
#         """
#         Args:
#             ctx: Context object used to calculate gradients.
#             points: FloatTensor of shape `(P, 3)`
#             points_first_idx: LongTensor of shape `(N,)` indicating the first point
#                 index in each example in the batch
#             tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
#                 triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
#             tris_first_idx: LongTensor of shape `(N,)` indicating the first face
#                 index in each example in the batch
#             max_points: Scalar equal to maximum number of points in the batch
#             min_triangle_area: (float, defaulted) Triangles of area less than this
#                 will be treated as points/lines.
#         Returns:
#             dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
#                 euclidean distance of `p`-th point to the closest triangular face
#                 in the corresponding example in the batch
#             idxs: LongTensor of shape `(P,)` indicating the closest triangular face
#                 in the corresponding example in the batch.

#             `dists[p]` is
#             `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
#             where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
#             face `(v0, v1, v2)`

#         """
#         dists, idxs = _C.point_face_dist_forward(
#             points,
#             points_first_idx,
#             tris,
#             tris_first_idx,
#             max_points,
#             min_triangle_area,
#         )
#         # ctx.save_for_backward(points, tris, idxs)
#         # ctx.min_triangle_area = min_triangle_area
#         return dists, idxs

#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         # 保存需要在反向传播中使用的上下文信息
#         points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area = inputs
#         dists, idxs = output
#         ctx.save_for_backward(points, tris, idxs)
#         ctx.mark_non_differentiable(idxs)
#         ctx.min_triangle_area = min_triangle_area

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_dists, grad_idxs):
#         # grad_dists = grad_dists.contiguous()
#         points, tris, idxs = ctx.saved_tensors
#         min_triangle_area = ctx.min_triangle_area
#         # print("check here backward", points, tris, idxs, grad_dists, min_triangle_area)
#         # print("check data attribute", points.data, tris.data, idxs.data, grad_dists.data, min_triangle_area)
#         grad_points, grad_tris = _C.point_face_dist_backward(
#             points, tris, idxs, grad_dists, min_triangle_area
#         )
#         return grad_points, None, grad_tris, None, None, None
    
#     @staticmethod
#     def vmap(info, in_dims, points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area):
#         # print("here is vmap", in_dims)
#         # print(points.shape, points_first_idx.shape, tris.shape, tris_first_idx.shape, max_points, min_triangle_area)
        
#         if in_dims[0] is not None:
#             dim_0, dim_1, dim_2 = points.shape
#             points = points.reshape(-1, 3)

#         if in_dims[2] is not None:
#             tris = tris.reshape(-1, 3, 3)

#         dists, idxs = _PointFaceDistance.apply(points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area)

#         if in_dims[0] is not None:
#             dists = dists.reshape(dim_0, -1)
#             idxs = idxs.reshape(dim_0, -1)

#         return (dists, idxs), (0, 0)


# point_face_distance = _PointFaceDistance.apply


# # FacePointDistance
# class _FacePointDistance(Function):
#     """
#     Torch autograd Function wrapper FacePointDistance Cuda implementation
#     """
#     # generate_vmap_rule = True

#     @staticmethod
#     def forward(
#         points,
#         points_first_idx,
#         tris,
#         tris_first_idx,
#         max_tris,
#         min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
#     ):
#         """
#         Args:
#             ctx: Context object used to calculate gradients.
#             points: FloatTensor of shape `(P, 3)`
#             points_first_idx: LongTensor of shape `(N,)` indicating the first point
#                 index in each example in the batch
#             tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
#                 triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
#             tris_first_idx: LongTensor of shape `(N,)` indicating the first face
#                 index in each example in the batch
#             max_tris: Scalar equal to maximum number of faces in the batch
#             min_triangle_area: (float, defaulted) Triangles of area less than this
#                 will be treated as points/lines.
#         Returns:
#             dists: FloatTensor of shape `(T,)`, where `dists[t]` is the squared
#                 euclidean distance of `t`-th triangular face to the closest point in the
#                 corresponding example in the batch
#             idxs: LongTensor of shape `(T,)` indicating the closest point in the
#                 corresponding example in the batch.

#             `dists[t] = d(points[idxs[t]], tris[t, 0], tris[t, 1], tris[t, 2])`,
#             where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
#             face `(v0, v1, v2)`.
#         """
#         dists, idxs = _C.face_point_dist_forward(
#             points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
#         )
#         # ctx.save_for_backward(points, tris, idxs)
#         # ctx.min_triangle_area = min_triangle_area
#         return dists, idxs

#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         # 保存需要在反向传播中使用的上下文信息
#         points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area = inputs
#         dists, idxs = output
#         ctx.save_for_backward(points, tris, idxs)
#         ctx.mark_non_differentiable(idxs)
#         ctx.min_triangle_area = min_triangle_area

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_dists, grad_idxs):
#         grad_dists = grad_dists.contiguous()
        
#         points, tris, idxs = ctx.saved_tensors
#         min_triangle_area = ctx.min_triangle_area

#         grad_points, grad_tris = _C.face_point_dist_backward(
#             points, tris, idxs, grad_dists, min_triangle_area
#         )
#         return grad_points, None, grad_tris, None, None, None

#     @staticmethod
#     def vmap(info, in_dims, points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area):
#         if in_dims[0] is not None:
#             dim_0, dim_1, dim_2 = points.shape
#             points = points.reshape(-1, 3)

#         if in_dims[2] is not None:
#             tris = tris.reshape(-1, 3, 3)
#         dists, idxs = _FacePointDistance.apply(points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area)

#         if in_dims[0] is not None:
#             dists = dists.reshape(dim_0, -1)
#             idxs = idxs.reshape(dim_0, -1)

#         return (dists, idxs), (0, 0)


# face_point_distance = _FacePointDistance.apply


# def my_point_mesh_face_distance(
#     meshes: Meshes,
#     pcls: Pointclouds,
#     min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
# ):
#     """
#     Computes the distance between a pointcloud and a mesh within a batch.
#     Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
#     sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

#     `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
#         to the closest triangular face in mesh and averages across all points in pcl
#     `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
#         mesh to the closest point in pcl and averages across all faces in mesh.

#     The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
#     and then averaged across the batch.

#     Args:
#         meshes: A Meshes data structure containing N meshes
#         pcls: A Pointclouds data structure containing N pointclouds
#         min_triangle_area: (float, defaulted) Triangles of area less than this
#             will be treated as points/lines.

#     Returns:
#         loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
#             between all `(mesh, pcl)` in a batch averaged across the batch.
#     """

#     if len(meshes) != len(pcls):
#         raise ValueError("meshes and pointclouds must be equal sized batches")
#     N = len(meshes)

#     # packed representation for pointclouds
#     points = pcls.points_packed()  # (P, 3)
#     points_first_idx = pcls.cloud_to_packed_first_idx()
#     max_points = pcls.num_points_per_cloud().max().item()

#     # packed representation for faces
#     verts_packed = meshes.verts_packed()
#     faces_packed = meshes.faces_packed()
#     tris = verts_packed[faces_packed]  # (T, 3, 3)  
#     tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
#     max_tris = meshes.num_faces_per_mesh().max().item()

#     # 使用 vmap 向量化 point_face_distance 和 face_point_distance

#     # point_face_vmap = torch.vmap(point_face_distance, in_dims=(0, 0, 0, 0, None, None), randomness='same')
#     # face_point_vmap = torch.vmap(face_point_distance, in_dims=(0, 0, 0, 0, None, None), randomness='same')

#     # @torch.compile
#     # def point_face_vmap(a, b, c, d, e, f):
#     #     return torch.func.vmap(point_face_distance, in_dims=(0, 0, 0, 0, None, None), randomness='same')(a, b, c, d, e, f)
    
#     # @torch.compile
#     # def face_point_vmap(a, b, c, d, e, f):
#     #     return torch.func.vmap(face_point_distance, in_dims=(0, 0, 0, 0, None, None), randomness='same')(a, b, c, d, e, f)

#     # # point to face distance: shape (P,)
#     # point_to_face, _ = point_face_distance(
#     #     points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
#     # )

#     # point to face distance: shape (P,)
#     point_to_face, _ = point_face_distance(
#         points.unsqueeze(0), points_first_idx.unsqueeze(0), tris.unsqueeze(0), tris_first_idx.unsqueeze(0), max_points, min_triangle_area
#     )
#     point_to_face = point_to_face.reshape(-1)


#     # weight each example by the inverse of number of points in the example
#     point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
#     num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
#     weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
#     # print(weights_p.shape)
#     # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
#     weights_p = 1.0 / weights_p.float()
#     point_to_face = point_to_face * weights_p # shape(P,)
#     # point_dist = point_to_face.sum() / N

#     # face to point distance: shape (T,)
#     face_to_point, _ = face_point_distance(
#         points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
#     )

#     # # face to point distance: shape (T,)
#     # face_to_point, _ = face_point_distance(
#     #     points.unsqueeze(0), points_first_idx.unsqueeze(0), tris.unsqueeze(0), tris_first_idx.unsqueeze(0), max_tris, min_triangle_area
#     # )
#     # face_to_point = face_to_point.reshape(-1)

#     # weight each example by the inverse of number of faces in the example
#     tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
#     num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
#     weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
#     weights_t = 1.0 / weights_t.float()
#     face_to_point = face_to_point * weights_t
#     # face_dist = face_to_point.sum() / N

#     return point_to_face.reshape(N, -1), face_to_point.reshape(N, -1)