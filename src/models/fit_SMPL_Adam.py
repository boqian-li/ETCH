import trimesh
from smplx import SMPL
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from torch.nn import functional as F
from tqdm import tqdm
from utils.GT_utils import save_points_with_vector, save_points_with_color
from utils.prior import MaxMixturePrior
import numpy as np
import os
from pytorch3d.structures import Meshes, Pointclouds
# from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
# from utils.customized_losses import point_2_mesh_distance


def get_markers(args, inner_points, part_labels, confidences):
    '''
    inner_points: Tensor shape(B, K, 3)
    part_labels: Tensor shape(B, K)
    confidences: Tensor shape(B, K, 1)

    return:
    pred_markers_position: Tensor, shape(B, num_markers, 3)
    valid_mask: Tensor shape(B, num_markers)
    '''
    B = inner_points.shape[0]
    valid_mask = torch.zeros(B, len(args.markerset), dtype=torch.bool, device=inner_points.device)
    pred_markers_position = torch.zeros(B, len(args.markerset), 3, device=inner_points.device)
    for b in range(B):
        inner_points_ = inner_points[b]
        part_labels_ = part_labels[b]
        confidences_ = confidences[b]
        for label in range(len(args.markerset)):
            label_mask = part_labels_ == label
            if label_mask.sum() == 0:
                continue
            label_points_ = inner_points_[label_mask] # shape(unsure, 3)
            label_confidences_ = confidences_[label_mask] # shape(unsure, 1)


            top_k = min(label_mask.sum().item(), 3)

            _, indices = torch.topk(label_confidences_, top_k, dim=0, largest=True)
            indices = indices.squeeze()
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)

            top_k_points_ = label_points_[indices] # shape(topk, 3)
            top_k_confidences_ = label_confidences_[indices] # shape(topk, 1)

            # Make larger confidence positions have higher weight than the confidence weight itself
            alpha = 20
            weights = top_k_confidences_ ** alpha # 1.0 / top_k_confidences_
            weighted_sum = (top_k_points_ * weights).sum(dim=0) # shape(3,)
            total_weight = weights.sum()
            weighted_center = weighted_sum / total_weight # shape(3,)

            valid_mask[b, label] = True
            pred_markers_position[b, label] = weighted_center

    return pred_markers_position, valid_mask


def fit_smpl(args, inner_points, part_labels, confidences, steps_stage0=400, steps_stage1=800, lr=1e-2):
    '''
    Fit SMPL model

    args
    inner_points: Tensor shape(B, K, 3)
    part_labels: Tensor shape(B, K)
    confidences: Tensor shape(B, K, 1)
    '''
    
    B = inner_points.shape[0]

    ### local config: ###
    use_marker = True
    use_pose_prior = False
    use_mean_shape = False

    #####################
    
    # get pred markers position
    pred_markers_position, valid_mask = get_markers(args, inner_points, part_labels, confidences) # return: shape(B, num_markers, 3) shape(B, num_markers)

    # Initialize SMPL model
    model_path = 'datafolder/body_models/smpl/neutral/SMPL_NEUTRAL_10pc_rmchumpy.pkl'  
    smpl_model = SMPL(model_path=model_path, create_global_orient=False, create_body_pose=False,
                      create_betas=False, create_transl=False).to(args.device) 
    if use_pose_prior:
        pose_prior = MaxMixturePrior(prior_folder='datafolder/body_models/prior/', num_gaussians=8).to(args.device)

    loss_weights = {
        "marker_loss": 1.0 * 10 ** 0,
        "mean_shape_loss": 1e-2 * 10 ** 0,
        # "point_mesh_distance": 1.0,
        # "part_pmdistance": 1.0,
        "pose_prior_loss": 1e-7 * 10 ** 0,
    }
    
    # STAGE 0: ONLY OPTIMIZE POSE AND TOP BETAS
    print("Optimization stage 0:")

    # Initialize optimization variables
    pose = torch.nn.Parameter(torch.zeros((B, smpl_model.NUM_BODY_JOINTS * 3)).to(args.device), requires_grad=True)
    shape_optimized = torch.nn.Parameter(torch.zeros((B, 2)).to(args.device), requires_grad=True)
    shape_frozen = torch.nn.Parameter(torch.zeros((B, smpl_model.num_betas - 2)).to(args.device), requires_grad=False)
    global_orient = torch.nn.Parameter(torch.zeros((B, 3)).to(args.device), requires_grad=True)
    translation = torch.nn.Parameter(torch.zeros((B, 3)).to(args.device), requires_grad=True)

    # Get parameters to optimize
    parameters = [
        shape_optimized,
        global_orient,
        pose,
        translation
    ]

    # Use Adam optimizer
    optimizer = torch.optim.Adam(parameters, lr=lr)

    pbar_stage0 = tqdm(range(steps_stage0))
    for step in pbar_stage0:  
        optimizer.zero_grad()

        # forward
        smpl_output = smpl_model(global_orient=global_orient, body_pose=pose, betas=torch.cat([shape_optimized, shape_frozen], dim=1), transl=translation, return_verts=True) # torch.cat([pose_optimized, pose_unoptimized], dim=1)
        smpl_vertices = smpl_output.vertices # shape(B, V, 3)

        marker_vindices = torch.tensor(list(args.markerset.values()), device=smpl_vertices.device).unsqueeze(0).expand(B, -1)
        forwarded_markers_position = torch.gather(smpl_vertices, 1, marker_vindices.unsqueeze(-1).expand(-1, -1, 3)) # shape(B, num_markers, 3)

        # compute losses
        losses = {}

        # marker loss
        marker_loss = F.mse_loss(forwarded_markers_position[valid_mask], pred_markers_position[valid_mask])
        losses['marker_loss'] = marker_loss

        # mean shape loss
        if use_mean_shape:
            mean_shape_loss = torch.mean(shape_optimized ** 2)
            losses['mean_shape_loss'] = mean_shape_loss

        # pose prior loss
        if use_pose_prior:
            pose_prior_loss = pose_prior(pose, None)
            losses['pose_prior_loss'] = pose_prior_loss.sum()

        all_loss = 0.0
        for key in losses.keys():
            all_loss += loss_weights[key] * losses[key]

        all_loss.backward()
        optimizer.step()
        for k, v in losses.items():
            losses[k] = v.item()
        pbar_stage0.set_postfix(ordered_dict=losses)
    


    # STAGE 1: OPTIMIZE POSE AND ALL BETAS
    print("Optimization stage 1:")

    # Initialize optimization variables
    # pose = torch.nn.Parameter(pose.detach(), requires_grad=True)
    shape = torch.nn.Parameter(torch.cat([shape_optimized, shape_frozen], dim=1).detach(), requires_grad=True)
    # global_orient = torch.nn.Parameter(global_orient.detach(), requires_grad=True)
    # translation = torch.nn.Parameter(translation.detach(), requires_grad=True)


    # Get parameters to optimize
    parameters = [
        shape,
        global_orient,
        pose,
        translation
    ]

    # Use Adam optimizer
    optimizer = torch.optim.Adam(parameters, lr=lr)

    pbar_stage1 = tqdm(range(steps_stage1))
    for step in pbar_stage1:  
        optimizer.zero_grad()

        # forward
        smpl_output = smpl_model(global_orient=global_orient, body_pose=pose, betas=shape, transl=translation, return_verts=True) # torch.cat([pose_optimized, pose_unoptimized], dim=1)
        smpl_vertices = smpl_output.vertices # shape(B, V, 3)

        marker_vindices = torch.tensor(list(args.markerset.values()), device=smpl_vertices.device).unsqueeze(0).expand(B, -1)
        forwarded_markers_position = torch.gather(smpl_vertices, 1, marker_vindices.unsqueeze(-1).expand(-1, -1, 3)) # shape(B, num_markers, 3)

        # compute losses
        losses = {}

        # marker loss
        marker_loss = F.mse_loss(forwarded_markers_position[valid_mask], pred_markers_position[valid_mask])
        losses['marker_loss'] = marker_loss

        # mean shape loss
        if use_mean_shape:
            mean_shape_loss = torch.mean(shape ** 2)
            losses['mean_shape_loss'] = mean_shape_loss

        # pose prior loss
        if use_pose_prior:
            pose_prior_loss = pose_prior(pose, None)
            losses['pose_prior_loss'] = pose_prior_loss.sum()

        all_loss = 0.0
        for key in losses.keys():
            all_loss += loss_weights[key] * losses[key]

        all_loss.backward()
        optimizer.step()
        for k, v in losses.items():
            losses[k] = v.item()
        pbar_stage1.set_postfix(ordered_dict=losses)



    # get final smpl meshes
    final_mesh_list = []
    for b in range(B):
        final_smpl_mesh = trimesh.Trimesh(smpl_output.vertices[b].detach().cpu().numpy(), smpl_model.faces, process=False, maintain_order=True)
        final_mesh_list.append(final_smpl_mesh)
    
    return final_mesh_list, pred_markers_position, valid_mask


# def compute_marker_loss(args, forwarded_markers_position, pred_markers_position, valid_mask, stage=0):
#     '''
#     forwarded_markers_position: Tensor, shape(B, num_markers, 3)
#     pred_markers_position: Tensor, shape(B, num_markers, 3)
#     valid_mask: Tensor, shape(B, num_markers)
#     '''
#     losses = {}

#     if stage == 0:
#         # compute correspondence loss
        

#     # if stage == 1:
#     #     # compute s2m, m2c losses
#     #     # chamfer_loss, _ = chamfer_distance(pyt3d_smpl_mesh.verts_packed().unsqueeze(0), inner_points.unsqueeze(0))
#     #     # losses['chamfer_loss'] = chamfer_loss

#     #     correspondence_loss = F.mse_loss(anchored_points, inner_points)
#     #     losses['correspondence_loss'] = correspondence_loss

#     #     pyt3d_inner_points = Pointclouds(points=[inner_points])
#     #     point_mesh_distance = point_2_mesh_distance(pyt3d_smpl_mesh, pyt3d_inner_points)
#     #     losses['point_mesh_distance'] = point_mesh_distance

#     #     # compute per part loss
#     #     losses['part_pmdistance'] = 0.0
#     #     for label in range(len(args.all_seginfo['label_2_color'])):
#     #         if label == (len(args.all_seginfo['label_2_color']) - 1):
#     #             continue
#     #         label_mask = part_labels == label
#     #         if label_mask.sum() == 0:
#     #             continue
#     #         label_points = inner_points[label_mask]
#     #         pyt3d_label_points = Pointclouds(points=[label_points])

#     #         label_mesh_vertices = pyt3d_smpl_mesh.verts_packed()[args.label_2_verticesfilter_faces[label]["vertices_filter"]]
#     #         label_mesh_faces = args.label_2_verticesfilter_faces[label]["per_label_filtered_faces"]
#     #         pyt3d_label_mesh = Meshes(verts=[label_mesh_vertices], faces=[label_mesh_faces])

#     #         losses["part_pmdistance"] += point_2_mesh_distance(pyt3d_label_mesh, pyt3d_label_points)
        

#     # other losses??? TODO


#     return losses

if __name__ == "__main__":
    pass