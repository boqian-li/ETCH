from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs, _handle_pointcloud_input
import torch
import numpy as np
import tqdm
from typing import Optional, Union


class OptimizationSMPL(torch.nn.Module):
    def __init__(self):
        super(OptimizationSMPL, self).__init__()

        self.pose = torch.nn.Parameter(torch.zeros(1, 72).cuda())
        self.beta = torch.nn.Parameter((torch.zeros(1, 10).cuda()))
        self.trans = torch.nn.Parameter(torch.zeros(1, 3).cuda())
        # self.scale = torch.nn.Parameter(torch.ones(1).cuda()*1)

    def forward(self):
        return self.pose, self.beta, self.trans


## Utility Functions for Chamfer Distance
def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_normals,
    y_normals,
    weights,
    batch_reduction: Union[str, None],
    point_reduction: Union[str, None],
    norm: int,
    abs_cosine: bool,
):
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    if point_reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped

        if batch_reduction is not None:
            # batch_reduction == "sum"
            cham_x = cham_x.sum()
            if return_normals:
                cham_norm_x = cham_norm_x.sum()
            if batch_reduction == "mean":
                div = weights.sum() if weights is not None else max(N, 1)
                cham_x /= div
                if return_normals:
                    cham_norm_x /= div

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
        single_directional: If False (default), loss comes from both the distance between
            each point in x and its nearest neighbor in y and each point in y and its nearest
            neighbor in x. If True, loss is the distance between each point in x and its
            nearest neighbor in y.
        abs_cosine: If False, loss_normals is from one minus the cosine similarity.
            If True (default), loss_normals is from one minus the absolute value of the
            cosine similarity, which means that exactly opposite normals are considered
            equivalent to exactly matching normals, i.e. sign does not matter.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_normals,
        y_normals,
        weights,
        batch_reduction,
        point_reduction,
        norm,
        abs_cosine,
    )
    if single_directional:
        return cham_x, cham_norm_x
    else:
        cham_y, cham_norm_y = _chamfer_distance_single_direction(
            y,
            x,
            y_lengths,
            x_lengths,
            y_normals,
            x_normals,
            weights,
            batch_reduction,
            point_reduction,
            norm,
            abs_cosine,
        )
        if point_reduction is not None:
            return (
                cham_x + cham_y,
                (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None,
            )
        return (
            (cham_x, cham_y),
            (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None,
        )



class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
        bidirectional: Optional[bool] = False,
        reverse: Optional[bool] = False,
        batch_reduction: Optional[str] = "mean",
        point_reduction: Optional[str] = "sum",
    ):
        if reverse:
            _source = target_cloud
            _target = source_cloud
        else:
            _source = source_cloud
            _target = target_cloud
        return chamfer_distance(
            _source, _target,
            single_directional= not bidirectional,
            batch_reduction=batch_reduction, 
            point_reduction=point_reduction)[0]

def fit_cham(SMPL_model, pred_mesh, vertices_scan, prior,init, bidir=0):
    chamferDist = ChamferDistance()
    parameters_smpl = OptimizationSMPL().cuda()
    parameters_smpl.pose = init['pose']
    parameters_smpl.beta = init['beta']
    parameters_smpl.trans = init['trans']
    parameters_smpl.scale = init['scale']
    
    lr = 2e-2
    
    optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=lr)
    iterations = 500
    ind_verts = np.arange(6890)
    # pred_mesh_torch = torch.FloatTensor(pred_mesh).cuda()

    factor_beta_reg = 0.2

    for i in tqdm.tqdm(range(iterations),desc="Chamfer"):
        pose, beta, trans, scale = parameters_smpl.forward()
        #beta = beta*3
        vertices_smpl = (SMPL_model.forward(body_pose=pose[:, 3:], betas=beta, global_orient=pose[:, :3], return_verts=True).vertices[0] + trans)*scale
        # distances = torch.abs(pred_mesh_torch - vertices_smpl)

        if bidir==0:
            d1 = torch.sqrt(chamferDist(torch.FloatTensor(vertices_scan).cuda().unsqueeze(0), vertices_smpl.unsqueeze(0), False)).mean()
            d2 = torch.sqrt(chamferDist(vertices_smpl.unsqueeze(0), torch.FloatTensor(vertices_scan).cuda().unsqueeze(0), False)).mean()

            loss = d1 + d2
        elif bidir==1: ## Partial
            loss = torch.sqrt(chamferDist(torch.FloatTensor(vertices_scan).cuda().unsqueeze(0), vertices_smpl.unsqueeze(0), False)).mean()
        elif bidir==-1: ##Clutter
            loss = torch.sqrt(chamferDist(vertices_smpl.unsqueeze(0), torch.FloatTensor(vertices_scan).cuda().unsqueeze(0), False)).mean()

        prior_loss = prior.forward(pose[:, 3:], beta)
        beta_loss = (beta**2).mean()
        loss = loss + prior_loss*0.00000001 + beta_loss*factor_beta_reg

        optimizer_smpl.zero_grad()
        loss.backward()
        optimizer_smpl.step()
        
        for param_group in optimizer_smpl.param_groups:
            param_group['lr'] = lr*(iterations-i)/iterations

    with torch.no_grad():
        pose, beta, trans, scale = parameters_smpl.forward()
        #beta = beta*3
        vertices_smpl = (SMPL_model.forward(body_pose=pose[:, 3:], betas=beta, global_orient=pose[:, :3], return_verts=True).vertices[0] + trans)*scale
        pred_mesh3 = vertices_smpl.cpu().data.numpy()
        joints = SMPL_model.forward(body_pose=pose[:, 3:], betas=beta, global_orient=pose[:, :3], return_joints=True).joints[0]
        params = {}
        params['loss'] = loss 
        params['beta'] = beta 
        params['pose'] = pose 
        params['trans'] = trans 
        params['scale'] = scale
        params['joints'] = joints
    return pred_mesh3, params