from ..external.smplx.smplx import SMPL
from ..external.smplx.smplx.utils import Struct
from ..external.RobustSkinWeightsTransferCode.src.utils import (
    inpaint,
    smooth,
    find_matches_closest_surface,
)
import torch
import numpy as np
import trimesh
import igl
import os
import pickle


def compute_face_edge_lengths_and_areas(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    # Heron's formula
    s = (e0 + e1 + e2) / 2
    area = np.sqrt(np.clip(s * (s - e0) * (s - e1) * (s - e2), 0, None))
    edge_lengths = np.stack([e0, e1, e2], axis=1)
    return edge_lengths, area


def filter_mesh(mesh_new, mesh_raw):
    raw_v = mesh_raw.vertices
    raw_f = mesh_raw.faces
    new_v = mesh_new.vertices
    new_f = mesh_new.faces

    raw_edge_lengths, raw_areas = compute_face_edge_lengths_and_areas(raw_v, raw_f)
    new_edge_lengths, new_areas = compute_face_edge_lengths_and_areas(new_v, new_f)

    new_edge_ratio = new_edge_lengths / (raw_edge_lengths + 1e-8)
    new_area_ratio = new_areas / (raw_areas + 1e-8)

    edge_ratio_min, edge_ratio_max = 0.3, 2.0
    area_ratio_min, area_ratio_max = 0.1, 4.0

    def get_valid_face_mask(edge_ratio, area_ratio):

        edge_valid = np.all(
            (edge_ratio > edge_ratio_min) & (edge_ratio < edge_ratio_max), axis=1
        )
        area_valid = (area_ratio > area_ratio_min) & (area_ratio < area_ratio_max)
        return edge_valid & area_valid

    new_valid_mask = get_valid_face_mask(new_edge_ratio, new_area_ratio)

    new_f_filtered = new_f[new_valid_mask]

    new_v_filtered, new_f_filtered, _, _ = igl.remove_unreferenced(
        new_v, new_f_filtered
    )
    new_mesh_filtered = trimesh.Trimesh(
        vertices=new_v_filtered, faces=new_f_filtered, process=False
    )
    new_mesh_filtered.export("scan_mesh_new_filtered.obj")


def clean_mesh_numpy(mesh, area_eps=1e-12):

    V = mesh.vertices
    F = mesh.faces

    # 1. remove degenerate triangles
    degenerate_mask = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    F1 = F[~degenerate_mask]

    # 2. remove zero area triangles
    v0 = V[F1[:, 0]]
    v1 = V[F1[:, 1]]
    v2 = V[F1[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area_mask = area >= area_eps
    F2 = F1[area_mask]

    # 3. remove duplicate triangles
    F2_sorted = np.sort(F2, axis=1)
    _, unique_idx = np.unique(F2_sorted, axis=0, return_index=True)
    F3 = F2[sorted(unique_idx)]

    # 4. remove unused vertices, and remap face indices
    used_verts = np.unique(F3.flatten())
    old_to_new = -np.ones(V.shape[0], dtype=int)
    old_to_new[used_verts] = np.arange(len(used_verts))
    V_clean = V[used_verts]
    F_clean = old_to_new[F3]

    cleaned_mesh = trimesh.Trimesh(vertices=V_clean, faces=F_clean, process=False)
    return cleaned_mesh


def weights_transfer(source_mesh, target_mesh, lbs_weights):

    V, F = source_mesh.vertices, source_mesh.faces
    V1, F1, _, _ = igl.remove_unreferenced(V, F)
    if V.shape[0] != V1.shape[0]:
        print(f"[Warning] Source mesh has unreferenced vertices which were removed")
    N1 = igl.per_vertex_normals(V1, F1)
    V, F = target_mesh.vertices, target_mesh.faces
    V2, F2, _, _ = igl.remove_unreferenced(V, F)
    if V.shape[0] != V2.shape[0]:
        print(f"[Warning] Target mesh has unreferenced vertices which were removed")
    N2 = igl.per_vertex_normals(V2, F2)
    W = lbs_weights
    dDISTANCE_THRESHOLD = 0.05 * igl.bounding_box_diagonal(V2)  # threshold distance D
    dDISTANCE_THRESHOLD_SQRD = dDISTANCE_THRESHOLD * dDISTANCE_THRESHOLD
    dANGLE_THRESHOLD_DEGREES = 30  # threshold angle theta in degrees

    Matched, SkinWeights_interpolated = find_matches_closest_surface(
        V1, F1, N1, V2, F2, N2, W, dDISTANCE_THRESHOLD_SQRD, dANGLE_THRESHOLD_DEGREES
    )

    InpaintedWeights, success = inpaint(V2, F2, SkinWeights_interpolated, Matched)

    return InpaintedWeights, success


def animate():
    model_path = "datafolder/body_models/smpl/neutral/SMPL_NEUTRAL_10pc_rmchumpy.pkl"
    smpl_model = SMPL(
        model_path=model_path,
    )
    data_struct = Struct(**pickle.load(open(model_path, "rb"), encoding="latin1"))
    lbs_weights = data_struct.weights

    smpl_data_raw = torch.load("path to raw smpl params")
    new_pose = torch.load("path to new smpl params")

    transl = smpl_data_raw["transl"].cpu().numpy()
    smpl_output_raw, trans_mats_raw = smpl_model(
        betas=smpl_data_raw["betas"],
        global_orient=smpl_data_raw["global_orient"],
        body_pose=smpl_data_raw["body_pose"].reshape(-1, 69),
        transl=smpl_data_raw["transl"],
    )

    smpl_output_new, trans_mats_new = smpl_model(
        betas=smpl_data_raw["betas"],
        global_orient=smpl_data_raw["global_orient"],
        body_pose=new_pose["body_pose"].reshape(-1, 69),
        transl=smpl_data_raw["transl"],
    )
    smpl_mesh_new = trimesh.Trimesh(
        smpl_output_new.vertices.squeeze(0).detach().cpu().numpy(), smpl_model.faces
    )
    smpl_mesh_new.export("smpl_mesh_new.obj")

    smpl_mesh_raw = trimesh.load("path to raw smpl mesh")
    scan_mesh_raw = trimesh.load("path to raw scan mesh")
    scan_mesh_verts_raw = scan_mesh_raw.vertices

    scan_mesh_raw = clean_mesh_numpy(scan_mesh_raw)

    InpaintedWeights, success = weights_transfer(
        smpl_mesh_raw, scan_mesh_raw, lbs_weights
    )

    # Normalize weights
    sums = InpaintedWeights.sum(axis=1, keepdims=True)
    zero_rows = sums[:, 0] < 1e-12
    InpaintedWeights[zero_rows, 0] = 1.0  # give dummy weights to all-zero rows
    sums[zero_rows] = 1.0
    InpaintedWeights /= sums

    scan_mesh_verts_raw_untrans = (
        torch.from_numpy(scan_mesh_verts_raw).to(torch.float32) - transl
    )

    T = (
        torch.from_numpy(InpaintedWeights).float()
        @ trans_mats_raw.squeeze(0).view(24, 16)
    ).reshape(-1, 4, 4)

    # Check singular matrices
    det = torch.det(T)
    problematic = det.abs() < 1e-10
    T[problematic] = torch.eye(4)  # use identity matrix or neighborhood average

    num_verts = scan_mesh_verts_raw.shape[0]
    ones = torch.ones((num_verts, 1), dtype=scan_mesh_verts_raw_untrans.dtype)
    scan_mesh_verts_untrans_homo = torch.cat([scan_mesh_verts_raw_untrans, ones], dim=1)

    T_inv = torch.inverse(T)

    scan_mesh_verts_rest_homo = torch.matmul(
        T_inv, scan_mesh_verts_untrans_homo.unsqueeze(-1)
    ).squeeze(-1)

    T = torch.matmul(InpaintedWeights, trans_mats_new.squeeze(0).view(24, 16)).view(
        num_verts, 4, 4
    )

    scan_mesh_verts_new_homo = torch.matmul(
        T, scan_mesh_verts_rest_homo.unsqueeze(-1)
    ).squeeze(-1)

    scan_mesh_verts_new = scan_mesh_verts_new_homo[:, :3]
    scan_mesh_new = trimesh.Trimesh(
        scan_mesh_verts_new.detach().cpu().numpy() + transl, scan_mesh_raw.faces
    )
    scan_mesh_new.export("scan_mesh_new.obj")
    filter_mesh(scan_mesh_new, scan_mesh_raw)


if __name__ == "__main__":
    animate()
