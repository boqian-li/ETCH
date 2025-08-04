import numpy as np
import trimesh
from scipy.spatial import KDTree
import argparse
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import torch

def save_points_with_color(points, colors, filename):
    '''
    points: numpy array, shape(N, 3)
    colors: numpy array, shape(N, 3)
    '''
    # Ensure color values are between 0 and 1
    # colors = colors / 255.0

    # Create trimesh point cloud object
    point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors)

    # Save point cloud to file
    point_cloud.export(filename)

def save_points_with_vector(hit_points, vectors, file_path):
    num_points = len(hit_points)
    num_vectors = len(vectors)
    assert num_points == num_vectors
    # hit_points = np.stack(hit_points, axis=0) # shape(N, 3)
    # vectors = np.stack(vectors, axis=0)
    
    # Calculate vector end points
    vector_end_points = hit_points - vectors

    with open(file_path, 'w') as f:
        # Write header information
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points * 2}\n")  # Point cloud and vector end points
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {num_vectors}\n")  # Vectors (as line segments)
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        # Write point cloud and vector end points (point cloud in red, vector end points in blue)
        for p in hit_points:
            f.write(f"{p[0]} {p[1]} {p[2]} 255 0 0\n")  # Red
        for v in vector_end_points:
            f.write(f"{v[0]} {v[1]} {v[2]} 0 0 255\n")  # Blue
        
        # Write vectors (as line segments)
        for i in range(num_vectors):
            f.write(f"{i} {num_points + i}\n")

def random_rotate_point_cloud(batch_data_point_cloud: torch.Tensor):
    assert 0==1, "this is a test, please ignore it; should rotate in global orientation"
    """ Randomly rotate the point clouds to augment the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    B, N, _ = batch_data_point_cloud.shape

    # Generate random rotation angles
    angles = torch.rand(B, 3) * torch.pi

    # Generate rotation matrices
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # Generate batch rotation matrices
    Rx = torch.stack([
        torch.tensor([1, 0, 0], dtype=batch_data_point_cloud.dtype, device=batch_data_point_cloud.device).repeat(B, 1),
        torch.stack([torch.zeros(B, dtype=batch_data_point_cloud.dtype, device=batch_data_point_cloud.device), cos_vals[:, 0], -sin_vals[:, 0]], dim=1),
        torch.stack([torch.zeros(B, dtype=batch_data_point_cloud.dtype, device=batch_data_point_cloud.device), sin_vals[:, 0], cos_vals[:, 0]], dim=1)
    ], dim=1)

    Ry = torch.stack([
        torch.stack([cos_vals[:, 1], torch.zeros(B, dtype=batch_data_point_cloud.dtype, device=batch_data_point_cloud.device), sin_vals[:, 1]], dim=1),
        torch.tensor([0, 1, 0], dtype=batch_data_point_cloud.dtype, device=batch_data_point_cloud.device).repeat(B, 1),
        torch.stack([-sin_vals[:, 1], torch.zeros(B, dtype=batch_data_point_cloud.dtype, device=batch_data_point_cloud.device), cos_vals[:, 1]], dim=1)
    ], dim=1)

    Rz = torch.stack([
        torch.stack([cos_vals[:, 2], -sin_vals[:, 2], torch.zeros(B, dtype=batch_data_point_cloud.dtype, device=batch_data_point_cloud.device)], dim=1),
        torch.stack([sin_vals[:, 2], cos_vals[:, 2], torch.zeros(B, dtype=batch_data_point_cloud.dtype, device=batch_data_point_cloud.device)], dim=1),
        torch.tensor([0, 0, 1], dtype=batch_data_point_cloud.dtype, device=batch_data_point_cloud.device).repeat(B, 1)
    ], dim=1)

    # Calculate final rotation matrix
    rotation_matrix = torch.bmm(Rz, torch.bmm(Ry, Rx))

    # Apply rotation matrix in batch
    rotated_data = torch.bmm(batch_data_point_cloud, rotation_matrix)

    return rotated_data