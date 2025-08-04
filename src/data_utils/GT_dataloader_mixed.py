import torch
from torch.utils.data import Dataset
import os
import numpy as np
import trimesh
from tqdm import tqdm
import pickle
from scipy.spatial import cKDTree
import json
import potpourri3d as pp3d
from utils.GT_utils import save_points_with_vector
from smplx import SMPL
from typing import NewType
import random
from smplx import SMPLHLayer

Tensor = NewType("Tensor", torch.Tensor)


# def barycentric_interpolation(val, coords):
#     """
#     :param val: verts x 3 x d input matrix
#     :param coords: verts x 3 barycentric weights array
#     :return: verts x d weighted matrix
#     """
#     t = val * coords[..., np.newaxis]
#     ret = t.sum(axis=1)
#     return ret
def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Calculates the rotation matrices for a batch of rotation vectors
    Parameters
    ----------
    rot_vecs: torch.tensor Nx3
        array of N axis-angle vectors
    Returns
    -------
    R: torch.tensor Nx3x3
        The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(
        (batch_size, 3, 3)
    )

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def generate_random_rotation_matrix():
    # Generate a random axis-angle vector
    random_axis_angle = torch.rand(3) * 2 * torch.pi - torch.pi

    # Convert axis-angle to a rotation matrix
    rotation_matrix = rodrigues(random_axis_angle)

    return rotation_matrix


def rodrigues(axis_angle):
    # Compute the rotation matrix from an axis-angle vector
    theta = torch.norm(axis_angle)
    axis = axis_angle / (theta + 1e-6)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Create the skew-symmetric cross-product matrix
    K = torch.tensor(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    identity = torch.eye(3)
    rotation_matrix = identity + sin_theta * K + (1 - cos_theta) * torch.mm(K, K)

    return rotation_matrix


def subdivide(mesh):
    # Note: This function concatenates the newly generated vertices to the end of the original vertices,
    # so the indices of the original vertices remain unaffected. However, please manually verify this
    # behavior to ensure that the version of trimesh you are using indeed performs this operation.
    v_, f_ = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    new_mesh = trimesh.Trimesh(v_, f_)
    return new_mesh


def convert_geodesic_distances_to_confidence(geodesic_distances):
    sharpness_factor = 10.0  # Factor controlling sharpness
    exponential_values = np.exp(-sharpness_factor * geodesic_distances)
    return exponential_values


class GTDataset(Dataset):
    def __init__(self, args):

        self.seed = args.seed
        self.scan_dir = args.scan_dir
        self.smpl_dir = args.smpl_dir
        self.infopoints_dir = args.infopoints_dir
        activated_ids_path = args.activated_ids_path
        with open(activated_ids_path, "rb") as f:
            self.activated_ids = pickle.load(f)
        self.id_list = [
            id
            for id in os.listdir(self.scan_dir)
            if os.path.isdir(os.path.join(self.scan_dir, id))
            and os.path.isdir(os.path.join(self.smpl_dir, id))
            and os.path.isfile(os.path.join(self.infopoints_dir, f"{id}.npz"))
            and id in self.activated_ids
        ]
        self.id_list = sorted(self.id_list)
        self.num_point = args.num_point
        self.markerset = args.markerset
        self.aug = args.aug

        self.gender_dict = {0: "female", 1: "male"}
        # # select for parallel running
        # current_select = 3
        # select_list = [(0, 250), (250, 500), (500, 750), (750, len(self.id_list))]
        # self.id_list = self.id_list[select_list[current_select][0]:select_list[current_select][1]]
        # print(f"===== current select: {current_select}, id_list_len: {len(self.id_list)} =====")

        print(f"Num of data: {len(self.id_list)}")

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        item = {}

        id = self.id_list[index]

        infopoints_path = os.path.join(self.infopoints_dir, f"{id}.npz")
        scan_path = os.path.join(self.scan_dir, id, f"{id}.obj")
        smpl_path = os.path.join(self.smpl_dir, id, f"mesh_smpl_{id}.obj")
        smpl_info_path = os.path.join(self.smpl_dir, id, f"info_{id}.npz")
        # print(f"infopoints_path: {infopoints_path}")
        # print(f"scan_path: {scan_path}")
        # print(f"smpl_path: {smpl_path}")
        # print(f"smpl_info_path: {smpl_info_path}")
        # print(os.path.isfile(infopoints_path))
        # print(os.path.isfile(scan_path))
        # print(os.path.isfile(smpl_path))
        # print(os.path.isfile(smpl_info_path))
        assert (
            os.path.isfile(infopoints_path)
            and os.path.isfile(scan_path)
            and os.path.isfile(smpl_path)
            and os.path.isfile(smpl_info_path)
        )

        infopoints_data = np.load(infopoints_path)
        info_points = infopoints_data["info_points"]
        info_vectors = infopoints_data["info_vectors"]

        scan_mesh = trimesh.load_mesh(scan_path, process=False, maintain_order=True)
        smpl_mesh = trimesh.load_mesh(smpl_path, process=False, maintain_order=True)

        # 1. Center scan_mesh and smpl_mesh, for mesh only need to center info_points
        scan_vertices = scan_mesh.vertices
        scan_min_xyz = np.min(scan_vertices, axis=0)
        scan_max_xyz = np.max(scan_vertices, axis=0)
        scan_center = (scan_min_xyz + scan_max_xyz) / 2.0

        scan_mesh.vertices = scan_vertices - scan_center
        smpl_mesh.vertices = smpl_mesh.vertices - scan_center

        info_points = info_points - scan_center
        if self.aug:
            # 2. Randomly rotate scan_mesh and smpl_mesh around y-axis at origin, simultaneously rotate info_points and info_vectors
            # Generate a random rotation angle around y-axis
            theta = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            R = np.array(
                [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
            )
            # Rotate scan_mesh, smpl_mesh, info_points, info_vectors
            scan_mesh.vertices = scan_mesh.vertices @ R.T
            smpl_mesh.vertices = smpl_mesh.vertices @ R.T
            info_points = info_points @ R.T
            info_vectors = info_vectors @ R.T

        sample_points, _ = trimesh.sample.sample_surface(
            scan_mesh, self.num_point, seed=self.seed + 15
        )  # shape(num_point, 3)

        ## VECTORS
        # precompute for condition 1
        tree = cKDTree(info_points)
        dists, indices = tree.query(sample_points, k=1)

        # precompute for condition 2
        closest_points, distances, closest_faces = trimesh.proximity.closest_point(
            smpl_mesh, sample_points
        )

        # For each sample point, the corresponding vector either uses the vector from the nearest info point that meets the threshold, or is calculated from the closest point on smpl_mesh
        threshold = 0.01
        vectors = np.zeros((self.num_point, 3), dtype=np.float64)

        # condition 1:
        dists_condition = dists < threshold  # shape(num_point)
        vectors[dists_condition] = info_vectors[indices[dists_condition]]

        # condition 2:
        not_dists_condition = ~dists_condition
        vectors[not_dists_condition] = (
            sample_points[not_dists_condition] - closest_points[not_dists_condition]
        )

        ## GEODESIC DISTANCES AND LABELS
        new_smpl_mesh = subdivide(smpl_mesh)
        markers_indices = list(self.markerset.values())

        solver = pp3d.MeshHeatMethodDistanceSolver(
            new_smpl_mesh.vertices, new_smpl_mesh.faces
        )
        m_i_ = np.zeros((len(markers_indices), len(new_smpl_mesh.vertices)))
        for m, marker_index in enumerate(markers_indices):
            dist = solver.compute_distance(marker_index)
            m_i_[m] = dist

        geodesic_distances_on_vertices = np.min(
            m_i_, axis=0
        )  # shape(num_new_vertices, )
        labels_on_vertices = np.argmin(m_i_, axis=0)  # shape(num_new_vertices, )
        # markers_positions = new_smpl_mesh.vertices[
        #     markers_indices
        # ]  # shape(num_markers, 3)

        tree = cKDTree(new_smpl_mesh.vertices)
        inner_points = sample_points - vectors
        dists, indices = tree.query(inner_points, k=1)
        geodesic_distances = geodesic_distances_on_vertices[indices].reshape(-1, 1)
        labels = labels_on_vertices[indices]

        # convert geodesic_distances to confidence TODO
        confidences = convert_geodesic_distances_to_confidence(geodesic_distances)

        # smpl info
        smpl_info_data = np.load(smpl_info_path)

        item["id"] = id
        item["hitpts"] = torch.from_numpy(sample_points).type(
            dtype=torch.float32
        )  # shape(num_point, 3)
        item["vectors"] = torch.from_numpy(vectors).type(
            dtype=torch.float32
        )  # shape(num_point, 3)
        item["confidences"] = torch.from_numpy(confidences).type(
            dtype=torch.float32
        )  # shape(num_point, 1)
        item["labels"] = torch.from_numpy(labels).type(
            dtype=torch.long
        )  # shape(num_point)
        item["gender"] = self.gender_dict[smpl_info_data["gender"].item()]

        # for dynamic label and confidence
        # item['markers_positions'] = torch.from_numpy(markers_positions).type(dtype=torch.float32) # shape(num_markers, 3)

        # augmentation
        # if self.aug:
        #     raise ValueError("Are you sure to use augmentation?")
        #     Jtr = smpl_info_data["joints"]
        #     root_loc = torch.from_numpy(Jtr[0]).type(dtype=torch.float32)  # shape(3)

        #     # random rotation
        #     rotation_matrix = generate_random_rotation_matrix()  # shape(3, 3)
        #     item["hitpts"] = (
        #         torch.matmul(item["hitpts"] - root_loc, rotation_matrix.T) + root_loc
        #     )
        #     item["vectors"] = torch.matmul(item["vectors"], rotation_matrix.T)
        #     # item['markers_positions'] = torch.matmul(item['markers_positions'] - root_loc, rotation_matrix.T) + root_loc

        # # visualize:
        # export_path = f'./tmp/visual_{id}/'
        # os.makedirs(f'./tmp/visual_{id}/')
        # trimesh.PointCloud(item["hitpts"].cpu().numpy()).export(os.path.join(export_path, 'hitpts.obj'))
        # save_points_with_vector(item["hitpts"].cpu().numpy(), item['vectors'].cpu().numpy(), os.path.join(export_path, 'vectors.ply'))
        # trimesh.PointCloud(item["markers_positions"].cpu().numpy()).export(os.path.join(export_path, 'markers.obj'))

        return item
