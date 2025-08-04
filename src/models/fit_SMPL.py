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
# from utils.customized_losses import my_point_mesh_face_distance
import theseus as th


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





def fit_smpl(args, inner_points, part_labels, confidences, gender, steps_stage0=30, steps_stage1=50, lr_stage0=5e-1, lr_stage1=2e-1):
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
    # use_pose_prior = False
    # use_mean_shape = False
    use_point_mesh_distance = False

    #####################
    
    # get pred markers position
    pred_markers_position, valid_mask = get_markers(args, inner_points, part_labels, confidences) # return: shape(B, num_markers, 3) shape(B, num_markers)

    # Initialize SMPL model
    if gender == 'neutral':
        body_model_path = 'datafolder/body_models/smpl/neutral/SMPL_NEUTRAL_10pc_rmchumpy.pkl'
    elif gender == 'female':
        body_model_path = 'datafolder/body_models/smpl/female/SMPL_FEMALE_10pc.pkl'
    elif gender == 'male':
        body_model_path = 'datafolder/body_models/smpl/male/SMPL_MALE_10pc.pkl'
    else:
        raise ValueError(f'Unexpected gender: {gender}')
    smpl_model = SMPL(model_path=body_model_path, create_global_orient=False, create_body_pose=False,
                      create_betas=False, create_transl=False).to(args.device) 

    loss_weights = {
        "marker_loss": 1.0,
        # "mean_shape_loss": 1e-2 * 10 ** 0,
        "point_mesh_distance": 1.0 * 10 ** 2,
        # "part_pmdistance": 1.0,
        # "pose_prior_loss": 1e-7 * 10 ** 0,
    }

    def marker_error_fn_0(optim_vars, aux_vars):
        pose, shape_optimized, global_orient, translation = optim_vars
        pred_markers_position, valid_mask = aux_vars

        valid_mask = valid_mask.tensor.bool()

        batch_size = shape_optimized.tensor.shape[0]
        shape_frozen = torch.zeros((batch_size, smpl_model.num_betas - 2)).to(args.device)

        # forward
        smpl_output = smpl_model(global_orient=global_orient.tensor, body_pose=pose.tensor, betas=torch.cat([shape_optimized.tensor, shape_frozen], dim=1), transl=translation.tensor, return_verts=True)
        smpl_vertices = smpl_output.vertices # shape(B, V, 3)

        marker_vindices = torch.tensor(list(args.markerset.values()), device=smpl_vertices.device).unsqueeze(0).expand(batch_size, -1)
        forwarded_markers_position = torch.gather(smpl_vertices, 1, marker_vindices.unsqueeze(-1).expand(-1, -1, 3)) # shape(B, num_markers, 3)

        err = pred_markers_position.tensor - forwarded_markers_position
        masked_err = err * valid_mask.unsqueeze(-1)
        masked_err = masked_err.reshape(batch_size, -1)

        return masked_err
    
    def marker_error_fn_1(optim_vars, aux_vars):
        pose, shape, global_orient, translation = optim_vars
        pred_markers_position, valid_mask = aux_vars

        valid_mask = valid_mask.tensor.bool()

        batch_size = shape.tensor.shape[0]

        # forward
        smpl_output = smpl_model(global_orient=global_orient.tensor, body_pose=pose.tensor, betas=shape.tensor, transl=translation.tensor, return_verts=True)
        smpl_vertices = smpl_output.vertices # shape(B, V, 3)

        marker_vindices = torch.tensor(list(args.markerset.values()), device=smpl_vertices.device).unsqueeze(0).expand(batch_size, -1)
        forwarded_markers_position = torch.gather(smpl_vertices, 1, marker_vindices.unsqueeze(-1).expand(-1, -1, 3)) # shape(B, num_markers, 3)

        err = pred_markers_position.tensor - forwarded_markers_position
        masked_err = err * valid_mask.unsqueeze(-1)
        masked_err = masked_err.reshape(batch_size, -1)

        return masked_err


    
    
    # STAGE 0: ONLY OPTIMIZE POSE AND TOP BETAS
    print("Optimization stage 0:")

    # Initialize optimization variables
    pose = torch.zeros((B, smpl_model.NUM_BODY_JOINTS * 3)).to(args.device)
    shape_optimized = torch.zeros((B, 2)).to(args.device)
    global_orient = torch.zeros((B, 3)).to(args.device)
    translation = torch.zeros((B, 3)).to(args.device)

    pose = th.Vector(tensor=pose, name="pose")
    shape_optimized = th.Vector(tensor=shape_optimized, name="shape_optimized")
    global_orient = th.Vector(tensor=global_orient, name="global_orient")
    translation = th.Vector(tensor=translation, name="translation")

    pred_markers_position = th.Variable(tensor=pred_markers_position, name="pred_markers_position")
    valid_mask = th.Variable(tensor=valid_mask.float(), name="valid_mask")

    optim_vars = [pose, shape_optimized, global_orient, translation]
    aux_vars = [pred_markers_position, valid_mask]


    w_marker = th.ScaleCostWeight(loss_weights["marker_loss"])
    marker_cost_function = th.AutoDiffCostFunction(optim_vars, marker_error_fn_0, len(args.markerset) * 3, cost_weight=w_marker, aux_vars=aux_vars, name="marker_cost_function")

    

    objective = th.Objective().to(args.device)
    objective.add(marker_cost_function)
    
    optimizer = th.LevenbergMarquardt(objective, max_iterations=steps_stage0, step_size=lr_stage0)
    # optimizer = th.GaussNewton(objective, max_iterations=steps_stage0, step_size=lr_stage0)
    theseus_layer = th.TheseusLayer(optimizer).to(args.device)

    theseus_inputs = {
        "pose": pose,
        "shape_optimized": shape_optimized,
        "global_orient": global_orient,
        "translation": translation,

        "pred_markers_position": pred_markers_position,
        "valid_mask": valid_mask
    }

    updated_inputs, _ = theseus_layer.forward(theseus_inputs, optimizer_kwargs={"verbose": True, "damping": 0.01}) # TODO: damping = ??


    pose = updated_inputs["pose"]
    shape_optimized = updated_inputs["shape_optimized"]
    global_orient = updated_inputs["global_orient"]
    translation = updated_inputs["translation"]



    # STAGE 1: OPTIMIZE POSE AND ALL BETAS
    print("Optimization stage 1:")

    pose = pose.detach()
    shape_frozen = torch.zeros((B, smpl_model.num_betas - 2)).to(args.device)
    shape = torch.cat([shape_optimized, shape_frozen], dim=1).detach()
    global_orient = global_orient.detach()
    translation = translation.detach()

    pose = th.Vector(tensor=pose, name="pose")
    shape = th.Vector(tensor=shape, name="shape")
    global_orient = th.Vector(tensor=global_orient, name="global_orient")
    translation = th.Vector(tensor=translation, name="translation")


    optim_vars = [pose, shape, global_orient, translation]
    aux_vars = [pred_markers_position, valid_mask]


    w_marker = th.ScaleCostWeight(loss_weights["marker_loss"])
    marker_cost_function = th.AutoDiffCostFunction(optim_vars, marker_error_fn_1, len(args.markerset) * 3, cost_weight=w_marker, aux_vars=aux_vars, name="marker_cost_function")

    

    objective = th.Objective().to(args.device)
    objective.add(marker_cost_function)
    optimizer = th.LevenbergMarquardt(objective, max_iterations=steps_stage1, step_size=lr_stage1)
    theseus_layer = th.TheseusLayer(optimizer).to(args.device)

    theseus_inputs = {
        "pose": pose,
        "shape": shape,
        "global_orient": global_orient,
        "translation": translation,

        "pred_markers_position": pred_markers_position,
        "valid_mask": valid_mask
    }

    updated_inputs, _ = theseus_layer.forward(theseus_inputs, optimizer_kwargs={"verbose": True}) # TODO: damping = ??


    pose = updated_inputs["pose"]
    shape = updated_inputs["shape"]
    global_orient = updated_inputs["global_orient"]
    translation = updated_inputs["translation"]

    # get final smpl meshes
    smpl_output = smpl_model(global_orient=global_orient, body_pose=pose, betas=shape, transl=translation, return_verts=True)
    joints = smpl_output.joints # shape(B, J, 3)

    final_mesh_list = []
    for b in range(B):
        final_smpl_mesh = trimesh.Trimesh(smpl_output.vertices[b].detach().cpu().numpy(), smpl_model.faces, process=False, maintain_order=True)
        final_mesh_list.append(final_smpl_mesh)
    
    output_smpl_info = [pose.detach().cpu().numpy().reshape(B, 23, 3), shape.detach().cpu().numpy(), global_orient.detach().cpu().numpy(), translation.detach().cpu().numpy(), joints.detach().cpu().numpy()]
    # shape(B, 23, 3), shape(B, 10), shape(B, 3), shape(B, 3), shape(B, 45, 3)

    return final_mesh_list, pred_markers_position.tensor, valid_mask.tensor.bool(), output_smpl_info


if __name__ == "__main__":
    pass

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