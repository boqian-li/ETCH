# """
# Create a discrete voxel grid and spread SMPL function from surface to the volume.
# Author: Bharat
# Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
# """
# import os
# import numpy as np
# import pickle as pkl
# from os.path import exists, split, join
# from export_standardsmplhmesh import export_smplmesh
# import trimesh
# from psbody.mesh import Mesh
# import open3d as o3d
# import argparse

# def mesh_filter(mesh, vertex_2_part, condition_list):
#     # 创建一个布尔数组，标识需要保留的顶点
#     condition = np.array([vertex_2_part[i] not in condition_list for i in range(len(mesh.v if type(mesh) == Mesh else mesh.vertices))])

#     # 获取需要保留的顶点索引
#     valid_vertex_indices = np.where(condition)[0]

#     # 创建一个映射，从旧索引到新索引
#     index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_vertex_indices)}

#     # 过滤顶点
#     filtered_vertices = mesh.v[valid_vertex_indices] if type(mesh) == Mesh else mesh.vertices[valid_vertex_indices]

#     # 过滤面，保留所有顶点都在 valid_vertex_indices 中的面
#     filtered_faces = []
#     for face in (mesh.f) if type(mesh) == Mesh else (mesh.faces):
#         if all(vertex in index_map for vertex in face):
#             # 更新面中的顶点索引
#             new_face = [index_map[vertex] for vertex in face]
#             filtered_faces.append(new_face)

#     # 创建新的 mesh
#     filtered_mesh = trimesh.Trimesh(vertices=filtered_vertices, faces=filtered_faces, process=False, maintain_order=True)

#     return filtered_mesh

# def export_point_cloud_with_color(positions, colors, filename):
#     """
#     将点云位置和颜色导出为 PLY 文件。
#     :param positions: 形状为 (N, N, N, 3) 的 NumPy 数组，表示点云的位置。
#     :param colors: 形状为 (N, N, N, 3) 的 NumPy 数组，表示点云的颜色，值在 [-1, 1] 范围内。
#     :param filename: 导出的 PLY 文件名。
#     """
#     # 将位置和颜色数组展平为 (N^3, 3) 的形状
#     positions_flat = positions[:, :, positions.shape[0] // 2, :]
#     positions_flat = positions_flat.reshape(-1, 3)
#     colors_flat = colors[:, :, positions.shape[0] // 2, :]
#     colors_flat = colors_flat.reshape(-1, 3)

#     # 将颜色值从 [-1, 1] 映射到 [0, 1]
#     colors_flat = (colors_flat + 1) / 2

#     # 创建 Open3D 点云对象
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(positions_flat)
#     point_cloud.colors = o3d.utility.Vector3dVector(colors_flat)

#     # 将点云导出为 PLY 文件
#     o3d.io.write_point_cloud(filename, point_cloud)
#     print(f"Point cloud saved to {filename}")

# def barycentric_interpolation(val, coords):
#     """
#     :param val: verts x 3 x d input matrix
#     :param coords: verts x 3 barycentric weights array
#     :return: verts x d weighted matrix
#     """
#     t = val * coords[..., np.newaxis]
#     ret = t.sum(axis=1)
#     return ret


# def create_grid_pts(res=128):
#     x_ = np.linspace(-1, 1., res)
#     y_ = np.linspace(-1, 1., res)
#     z_ = np.linspace(-1, 1., res)

#     x, y, z = np.meshgrid(x_, y_, z_)
#     pts = np.concatenate([y.reshape(-1, 1), x.reshape(-1, 1), z.reshape(-1, 1)], axis=-1)
#     return pts


# # def transform_points(pts, scale, trans, reverse=False):
# #     if reverse:
# #         return (pts - trans)/scale
# #     return (pts * scale) + trans


# def main(args):
#     if not exists(args.save_dir):
#         os.makedirs(args.save_dir)

#     smpl_mesh = export_smplmesh()
#     smpl_mesh.export(f'{args.save_dir}/smpl_mesh_original.obj')
#     smpl_mesh = Mesh(smpl_mesh.vertices, smpl_mesh.faces)

#     # Bring SMPL mesh to [-1, 1]. Scaling such that height is 1.6m and center is 0
#     height = max(smpl_mesh.v.max(axis=0) - smpl_mesh.v.min(axis=0))
#     scale = TGT_HEIGHT / height
#     smpl_mesh.v *= scale

#     center = (smpl_mesh.v.max(axis=0) + smpl_mesh.v.min(axis=0))/2
#     center = TGT_CENTER - center
#     smpl_mesh.v += center

#     smpl_mesh_trimesh = trimesh.Trimesh(vertices=smpl_mesh.v, faces=smpl_mesh.f, process=False, maintain_order=True)
#     smpl_mesh_trimesh.export(f'{args.save_dir}/smpl_mesh_scaledtransformed.obj')

#     if args.skip_handsfeet:
#         smpl_mesh = mesh_filter(smpl_mesh, args.vertex_2_part, condition_list=["left_forearm", "right_forearm", "left_foot", "right_foot"])
#         smpl_mesh = Mesh(smpl_mesh.vertices, smpl_mesh.faces)
#         smpl_mesh_trimesh = trimesh.Trimesh(vertices=smpl_mesh.v, faces=smpl_mesh.f, process=False, maintain_order=True)
#         smpl_mesh_trimesh.export(f'{args.save_dir}/smpl_mesh_scaledtransformed_without_handsfeet.obj')


#     pts = create_grid_pts(res=args.res)  # shape: (res x res x res,  3); range: [-1, 1]
#     print("pts shape: ", pts.shape)
    
#     closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
#     vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))

#     # 将点云位置和颜色导出为 PLY 文件
#     export_point_cloud_with_color(pts.reshape(res, res, res, 3), closest_points.reshape(res, res, res, 3), 'data/correspondence_data_{}/closest_points.ply'.format(res))

#     # Check if interpolation is working as desired
#     assert np.allclose(closest_points, barycentric_interpolation(smpl_mesh.v[vert_ids], bary_coords), rtol=1e-05, atol=1e-08)
#     assert np.allclose(barycentric_interpolation(smpl_mesh.v[vert_ids], bary_coords), closest_points, rtol=1e-05, atol=1e-08)

#     '''Save the transformation'''
#     np.savez_compressed(join(args.save_dir, 'scale_center.npz'), scale=scale, center=center)
#     print('Saved scale_center', scale, center)

#     '''Save closest point'''
#     closest_points = closest_points.reshape(res, res, res, 3)
#     np.save(join(args.save_dir, 'closest_point.npy'), closest_points)
#     print('Saved closest_point', closest_points.shape)

#     '''Save distance to closest point'''
#     # save original values
#     closest_distance = ((pts - closest_points.reshape(-1, 3))**2).sum(axis=-1).reshape(res, res, res)**0.5
#     np.save(join(args.save_dir, 'closest_distance.npy'), closest_distance)
#     print('Saved closest_distance', closest_distance.shape)

#     print('Done')


# TGT_HEIGHT = 1.6
# TGT_CENTER = 0.
# if __name__ == "__main__":
#     res = 64
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--res', type=int, default=res)
#     parser.add_argument('--save_dir', type=str, default='data/correspondence_data_{}'.format(res))
#     parser.add_argument('--skip_handsfeet', type=bool, default=True)
#     parser.add_argument('--path_parts', default="data/useful_data/smpl_parts_dense.pkl", type=str)
#     args = parser.parse_args()

#     # Load smpl seginfo
#     with open(args.seg_info_path, 'rb') as f:
#         args.all_seginfo = pkl.load(f)


#     main(args)