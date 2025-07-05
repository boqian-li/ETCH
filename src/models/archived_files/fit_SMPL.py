# import trimesh
# from smplx import SMPL
# import torch
# from sklearn.linear_model import RANSACRegressor
# from sklearn.cluster import DBSCAN
# from torch.nn import functional as F
# from tqdm import tqdm
# from utils.GT_utils import save_points_with_vector, save_points_with_color
# from utils.prior import MaxMixturePrior
# import numpy as np
# import os
# from pytorch3d.structures import Meshes, Pointclouds
# # from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
# from utils.customized_losses import point_2_mesh_distance


# def barycentric_interpolation_torch(val, coords):
#     """
#     :param val: verts x 3 x d input matrix (torch.Tensor)
#     :param coords: verts x 3 barycentric weights array (torch.Tensor)
#     :return: verts x d weighted matrix (torch.Tensor)
#     """
#     # 在 PyTorch 中使用 unsqueeze 代替 np.newaxis
#     t = val * coords.unsqueeze(-1)
#     ret = t.sum(dim=1)  # 使用 dim 参数代替 axis
#     return ret


# def filter_points_by_clustering(args, inner_points, part_labels, eps=0.05, min_samples=5):
#     '''
#     inner_points: shape(M, 3)
#     part_labels: shape(M)
#     eps: float, DBSCAN 中的最大距离参数
#     min_samples: int, DBSCAN 中的最小样本数参数
#     '''

#     selected_indices = torch.zeros(inner_points.shape[0], dtype=torch.bool, device=inner_points.device)

#     for label in range(len(args.all_seginfo['label_2_color'])):
#         if label == (len(args.all_seginfo['label_2_color']) - 1):
#             continue

#         label_mask = part_labels == label
#         if label_mask.sum() == 0:
#             continue

#         label_points = inner_points[label_mask].detach().cpu().numpy()

#         # 使用 DBSCAN 聚类
#         clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(label_points)
#         labels = clustering.labels_

#         # 找到最大簇
#         unique_labels, counts = np.unique(labels, return_counts=True)
#         max_cluster_label = unique_labels[np.argmax(counts)]

#         # 选择最大簇中的点
#         if max_cluster_label != -1:  # -1 表示噪声点
#             max_cluster_indices = np.where(labels == max_cluster_label)[0]
#             # 将 max_cluster_indices 转换为全局索引
#             global_indices = torch.nonzero(label_mask, as_tuple=True)[0][max_cluster_indices]
#             # 更新 selected_indices
#             selected_indices[global_indices] = True

#     return torch.where(selected_indices)[0]

# def filter_points_by_correspondence(args, part_labels, vertices_indices):
#     '''
#     part_labels: Tensor shape(M)
#     vertices_indices: Tensor shape(M, 3)
#     '''
#     selected_indices = torch.zeros(vertices_indices.shape[0], dtype=torch.bool, device=vertices_indices.device)
#     for i in range(vertices_indices.shape[0]):
#         label = part_labels[i].item()

#         valid_indices = args.label_2_verticesfilter_faces[label]["vertices_filter"]
#         valid_indices_list = [args.reverse_index_map[v.item()] for v in valid_indices]
#         if all(vertices_indices[i][j].item() in valid_indices_list for j in range(3)):
#             selected_indices[i] = True

#     return torch.where(selected_indices)[0]
            




# def fit_smpl(args, inner_points, part_labels, vertices_indices, barycentric_coords, steps_stage0=800, steps_stage1=1200, lr=1e-2):
#     '''
#     Fit SMPL model to the predicted inner points, part labels, vertices indices, and barycentric coordinates.

#     args
#     inner_points: Tensor shape(M, 3)
#     part_labels: Tensor shape(M)
#     vertices_indices: Tensor shape(M, 3)
#     barycentric_coords: Tensor shape(M, 3)
#     all_seginfo: dict
#     '''

#     # 初始化 SMPL 模型
#     model_path = 'datafolder/body_models/smpl/neutral/SMPL_NEUTRAL_10pc_rmchumpy.pkl'  
#     smpl_model = SMPL(model_path=model_path, create_global_orient=False, create_body_pose=False,
#                       create_betas=False, create_transl=False).to(args.device) 
#     if args.use_prior:
#         prior = MaxMixturePrior(prior_folder='datafolder/useful_data_cape/prior/', num_gaussians=8).to(args.device)

#     loss_weights = {
#         "correspondence_loss": 1.0,
#         "point_mesh_distance": 1.0,
#         "part_pmdistance": 1.0,
#         "prior_loss": 1e-7,
#     }
    
#     # visualize correspondence between inner points and anchored points
#     # if 1 == 0:
#     #     # 通过 SMPL 模型计算顶点
#     #     smpl_output = smpl_model()
#     #     smpl_vertices = smpl_output.vertices.squeeze() # shape(V, 3)
#     #     vertices_coords = smpl_vertices[vertices_indices] # shape(M, 3, 3)
#     #     anchored_points = barycentric_interpolation_torch(vertices_coords, barycentric_coords) # shape(M, 3)
#     #     os.makedirs('data/vis', exist_ok=True)
#     #     save_points_with_vector(inner_points.detach().cpu().numpy(), inner_points.detach().cpu().numpy() - anchored_points.detach().cpu().numpy() - np.array([[1, 0, 0]]*inner_points.shape[0]), f'data/vis/{id_}_inner_points_error.ply')


#     # 初始化优化变量
#     ## 手部的pose不优化; 手腕的pose优化，但是算loss的时候它只会对前臂的扭曲造成影响，而手腕处的mesh本身并不参与 loss计算
#     pose_optimized = torch.nn.Parameter(torch.zeros((1, (smpl_model.NUM_BODY_JOINTS - 0) * 3)).to(args.device), requires_grad=True)
#     # pose_unoptimized = torch.nn.Parameter(torch.zeros((1, 2 * 3)).to(args.device), requires_grad=False)

#     shape = torch.nn.Parameter(torch.zeros((1, smpl_model.num_betas)).to(args.device), requires_grad=True)
#     global_orient = torch.nn.Parameter(torch.zeros((1, 3)).to(args.device), requires_grad=True)
#     translation = torch.nn.Parameter(torch.zeros((1, 3)).to(args.device), requires_grad=True)

#     # 获取需要优化的参数
#     parameters = [
#         shape,
#         global_orient,
#         pose_optimized,
#         translation
#     ]

#     # 使用 Adam 优化器
#     optimizer = torch.optim.Adam(parameters, lr=lr)

#     option = 0
#     # mode = args.mode
#     # if mode == "pred":
#     if option == 0:
#         print("Using clustering to filter inner points")
#         # 使用聚类筛选内点
#         inlier_points_indices = filter_points_by_clustering(args, inner_points, part_labels)
#         print(f"filter ratio: {inlier_points_indices.shape[0] / inner_points.shape[0]}")
#     elif option == 1:
#         assert 1 == 0
#         print("Using correspondence to filter inner points")
#         # 使用correspondence筛选内点
#         inlier_points_indices = filter_points_by_correspondence(args, part_labels, vertices_indices)
#         print(f"filter ratio: {inlier_points_indices.shape[0] / inner_points.shape[0]}")
#     else:
#         assert 1 == 0
#         inlier_points_indices = torch.arange(inner_points.shape[0]).to(inner_points.device)
#         print("Not using any filter")


        
#     inner_points = inner_points[inlier_points_indices]
#     part_labels = part_labels[inlier_points_indices]
#     vertices_indices = vertices_indices[inlier_points_indices]
#     barycentric_coords = barycentric_coords[inlier_points_indices]

#     # visualize filtered inner points with part labels
#     if option in [0, 1]:
#         gt_labels_colors = [np.array(args.all_seginfo["label_2_color"][label.item()], dtype=np.int64) for label in part_labels]
#         gt_labels_colors = np.stack(gt_labels_colors, axis=0) # shape(K, 3)
#         save_points_with_color(inner_points.detach().cpu().numpy(), gt_labels_colors, os.path.join(args.output_folder, f"{args.id_}", f"filtered_pred_inner_points_with_pred_part_labels_{args.id_}.ply"))


#     # Optimization stage 0:
#     print("Optimization stage 0:")
#     pbar_stage0 = tqdm(range(steps_stage0))
#     for step in pbar_stage0:  
#         optimizer.zero_grad()

#         # forward
#         smpl_output = smpl_model(global_orient=global_orient, body_pose=pose_optimized, betas=shape, transl=translation, return_verts=True) # torch.cat([pose_optimized, pose_unoptimized], dim=1)
#         smpl_vertices = smpl_output.vertices.squeeze() # shape(V, 3)
        
#         # get anchored points
#         vertices_coords = smpl_vertices[vertices_indices] # shape(M, 3, 3)
#         anchored_points = barycentric_interpolation_torch(vertices_coords, barycentric_coords) # shape(M, 3)

#         # get filtered pyt3d_smpl_mesh
#         pyt3d_smpl_mesh = Meshes(verts=[smpl_vertices[args.valid_vertex_indices]], faces=[args.filtered_faces])

#         losses = compute_loss(args, pyt3d_smpl_mesh, inner_points, part_labels, anchored_points, stage=0)
#         if args.use_prior:
#             prior_loss = prior(pose_optimized, None)
#             losses['prior_loss'] = prior_loss.sum()

#         all_loss = 0.0
#         for key in losses.keys():
#             all_loss += loss_weights[key] * losses[key]

#         all_loss.backward()
#         optimizer.step()
#         for k, v in losses.items():
#             losses[k] = v.item()
#         pbar_stage0.set_postfix(ordered_dict=losses)

#     # Optimization stage 1:
#     print("Optimization stage 1:")
#     pbar_stage1 = tqdm(range(steps_stage1))
#     for step in pbar_stage1:  
#         optimizer.zero_grad()

#         # forward
#         smpl_output = smpl_model(global_orient=global_orient, body_pose=pose_optimized, betas=shape, transl=translation, return_verts=True) # torch.cat([pose_optimized, pose_unoptimized], dim=1)
#         smpl_vertices = smpl_output.vertices.squeeze() # shape(V, 3)
        
#         # get anchored points
#         vertices_coords = smpl_vertices[vertices_indices] # shape(M, 3, 3)
#         anchored_points = barycentric_interpolation_torch(vertices_coords, barycentric_coords) # shape(M, 3)

#         # get filtered pyt3d_smpl_mesh
#         pyt3d_smpl_mesh = Meshes(verts=[smpl_vertices[args.valid_vertex_indices]], faces=[args.filtered_faces])

#         losses = compute_loss(args, pyt3d_smpl_mesh, inner_points, part_labels, anchored_points, stage=1)
#         if args.use_prior:
#             prior_loss = prior(pose_optimized, None)
#             losses['prior_loss'] = prior_loss.sum()

#         all_loss = 0.0
#         for key in losses.keys():
#             all_loss += loss_weights[key] * losses[key]

#         all_loss.backward()
#         optimizer.step()
#         for k, v in losses.items():
#             losses[k] = v.item()
#         pbar_stage1.set_postfix(ordered_dict=losses)

#     final_smpl_mesh = trimesh.Trimesh(smpl_output.vertices.squeeze().detach().cpu().numpy(), smpl_model.faces, process=False, maintain_order=True)
#     return final_smpl_mesh


# def compute_loss(args, pyt3d_smpl_mesh, inner_points, part_labels, anchored_points, stage=0):
#     '''
#     anchored_points: shape(M, 3)
#     pyt3d_smpl_mesh: pytorch3d.structures.Meshes
#     inner_points: shape(M, 3)
#     part_labels: shape(M)
#     anchored_points: shape(M, 3)
#     '''
#     losses = {}

#     if stage == 0:
#         # compute correspondence loss
#         correspondence_loss = F.mse_loss(anchored_points, inner_points)
#         losses['correspondence_loss'] = correspondence_loss

#     if stage == 1:
#         # compute s2m, m2c losses
#         # chamfer_loss, _ = chamfer_distance(pyt3d_smpl_mesh.verts_packed().unsqueeze(0), inner_points.unsqueeze(0))
#         # losses['chamfer_loss'] = chamfer_loss

#         correspondence_loss = F.mse_loss(anchored_points, inner_points)
#         losses['correspondence_loss'] = correspondence_loss

#         pyt3d_inner_points = Pointclouds(points=[inner_points])
#         point_mesh_distance = point_2_mesh_distance(pyt3d_smpl_mesh, pyt3d_inner_points)
#         losses['point_mesh_distance'] = point_mesh_distance

#         # compute per part loss
#         losses['part_pmdistance'] = 0.0
#         for label in range(len(args.all_seginfo['label_2_color'])):
#             if label == (len(args.all_seginfo['label_2_color']) - 1):
#                 continue
#             label_mask = part_labels == label
#             if label_mask.sum() == 0:
#                 continue
#             label_points = inner_points[label_mask]
#             pyt3d_label_points = Pointclouds(points=[label_points])

#             label_mesh_vertices = pyt3d_smpl_mesh.verts_packed()[args.label_2_verticesfilter_faces[label]["vertices_filter"]]
#             label_mesh_faces = args.label_2_verticesfilter_faces[label]["per_label_filtered_faces"]
#             pyt3d_label_mesh = Meshes(verts=[label_mesh_vertices], faces=[label_mesh_faces])

#             losses["part_pmdistance"] += point_2_mesh_distance(pyt3d_label_mesh, pyt3d_label_points)
        

#     # other losses??? TODO


#     return losses

# if __name__ == "__main__":
#     pass