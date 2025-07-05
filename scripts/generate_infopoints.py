import numpy as np
import trimesh
from scipy.spatial import KDTree
import argparse
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle as pkl
import random

def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret

def save_data(info_points, info_vectors, file_path):
    info_points = np.stack(info_points, axis=0)
    info_vectors = np.stack(info_vectors, axis=0)

    # save as npz
    np.savez(file_path, info_points=info_points, info_vectors=info_vectors)


def save_points_with_vector(args, hit_points, vectors, file_path):
    num_points = len(hit_points)
    num_vectors = len(vectors)
    assert num_points == num_vectors
    hit_points = np.stack(hit_points, axis=0)
    vectors = np.stack(vectors, axis=0)

    # ply_types = args.ply_types
    
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


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_data(args, model_path, smpl_path, id="abcd"):
    scan = trimesh.load_mesh(model_path, process=False, maintain_order=True)
    body = trimesh.load_mesh(smpl_path, process=False, maintain_order=True)
    # scan_vertices = scan.vertices

    points_on_body, face_indices = trimesh.sample.sample_surface_even(body, args.num_samples)

    triangle_vertices = body.faces[face_indices]  # shape(num_samples, 3)
    vertices_coords = body.vertices[triangle_vertices]  # shape(num_samples, 3, 3)
    barycentric_coords = trimesh.triangles.points_to_barycentric(vertices_coords, points_on_body)  # shape(num_samples, 3)

    assert np.allclose(points_on_body, barycentric_interpolation(vertices_coords, barycentric_coords), rtol=1e-05, atol=1e-08)
    assert np.allclose(barycentric_interpolation(vertices_coords, barycentric_coords), points_on_body, rtol=1e-05, atol=1e-08)
    
    # get normals
    vertex_normals = body.vertex_normals[triangle_vertices]  # shape: (num_samples, 3, 3)
    sampled_normals = barycentric_interpolation(vertex_normals, barycentric_coords)  # shape(num_samples, 3)
    sampled_normals = sampled_normals / np.linalg.norm(sampled_normals, axis=1, keepdims=True) # shape(num_samples, 3)



    
    # kdtree = KDTree(scan_vertices)
    # # Find nearest neighbor
    # _, nearest_vertex_indices = kdtree.query(abc, k=1, workers=-1)


    # min_bound, max_bound = scan.bounds
    # ray_length = np.linalg.norm(max_bound - min_bound).item() / 30  # max length of ray
    # ray_length_ops = ray_length / 5 # max length of ray in opposite direction
    ray_length = args.ray_length # 0.06
    ray_length_ops = args.ray_length_ops # 0.01
    # d_ = args.d_ # 1e-5 # a minimal distance for 0 tightness


    info_points = []
    info_vectors = []
    
    for i, sample_point in enumerate(points_on_body):

        normal = sampled_normals[i]

        ray_origin = sample_point
        ray_direction = normal
        
        # 1. intersection in normal direction
        locations, index_ray, index_tri = scan.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction],
            multiple_hits=False  
        )
        if len(locations) > 0:
            nearest_intersection = locations[0]
            distance = np.linalg.norm(nearest_intersection - sample_point)
            if distance < ray_length:
                # 2. intersection in opposite direction
                locations__, index_ray, index_tri = scan.ray.intersects_location(
                    ray_origins=[ray_origin],
                    ray_directions=[-ray_direction],
                    multiple_hits=False  
                )

                if len(locations__) > 0:
                    nearest_intersection__ = locations__[0]
                    distance__ = np.linalg.norm(nearest_intersection__ - sample_point)
                    if distance__ < ray_length_ops:
                        # print("intersection in opposite direction")
                        continue

                # 4. intersection between smpl parts
                locations___, index_ray, index_tri = body.ray.intersects_location(
                    ray_origins=[ray_origin],
                    ray_directions=[-ray_direction],
                    multiple_hits=False  
                )

                if len(locations___) > 0:
                    nearest_intersection___ = locations___[0]
                    distance___ = np.linalg.norm(nearest_intersection___ - sample_point)
                    if distance___ < 0.03:
                        # print("intersection between smpl parts")
                        continue
                

                locations_, index_ray, index_tri = body.ray.intersects_location(
                    ray_origins=[nearest_intersection],
                    ray_directions=[-ray_direction],
                    multiple_hits=False  
                )
                if not (len(locations_) > 0): 
                    # print(f"bad data")
                    continue
                if np.linalg.norm(locations_[0] - sample_point) < 1e-4:
                    # print("good data")
                    info_points.append(nearest_intersection)
                    info_vectors.append((nearest_intersection - sample_point))
                    continue
                else:
                    # print("else")
                    continue
                

        # # 2. intersection in opposite direction
        # locations, index_ray, index_tri = scan.ray.intersects_location(
        #     ray_origins=[ray_origin],
        #     ray_directions=[-ray_direction],
        #     multiple_hits=False  
        # )

        # if len(locations) > 0:
        #     nearest_intersection = locations[0]
        #     distance = np.linalg.norm(nearest_intersection - vertex)
        #     if distance < ray_length_ops:
        #         hit_points.append(nearest_intersection)
        #         vectors.append(ray_direction * d_)
        #         continue

        # # old 3. nearest neighbor point with normal direction
        # nearest_point = scan_vertices[nearest_vertex_indices[i]]
        # hit_points.append(nearest_point.view(np.ndarray))
        # vectors.append(ray_direction * d_)

        # 3. 
        # print("else")
        continue


    assert len(info_points) == len(info_vectors)
    if args.mode == 0:
        ply_path = os.path.join(args.folder_ply, f"{id}.ply")
        npz_path = os.path.join(args.folder_npz, f"{id}.npz")
    elif args.mode == 1:
        ply_path = f"{id}.ply"
        npz_path = f"{id}.npz"
    else:
        assert 1 == 0

    save_points_with_vector(args, info_points, info_vectors, file_path=ply_path)
    save_data(info_points, info_vectors, file_path=npz_path)
    # else:
        # print(f"skip id == {id}, len(hit_points) == {len(hit_points)}, len(vectors) == {len(vectors)}, len(correspondences) == {len(correspondences)}, len(labels) == {len(labels)}")

def main(args):
    print(f"\n ray_length={args.ray_length}, ray_length_ops={args.ray_length_ops}\n")

    if args.mode == 0:
        os.makedirs(args.folder_ply, exist_ok=True)
        os.makedirs(args.folder_npz, exist_ok=True)

        if args.parallel:
            # Get all data files from folders
            model_paths =[]
            smpl_paths = []
            ids = []
            id_list_model = [id for id in os.listdir(args.folder_model) if os.path.isdir(os.path.join(args.folder_model, id))]
            id_list_smpl = [id for id in os.listdir(args.folder_smpl) if os.path.isdir(os.path.join(args.folder_smpl, id))]
            id_list_final = list(set(id_list_model) & set(id_list_smpl))
            for id in id_list_final:
                model_path = os.path.join(args.folder_model, id, f"{id}.obj")
                smpl_path = os.path.join(args.folder_smpl, id, f"mesh_smpl_{id}.obj")

                model_paths.append(model_path)
                smpl_paths.append(smpl_path)
                ids.append(id)
            
            all_args = zip([args] * len(model_paths), model_paths, smpl_paths, ids)
            all_args = list(all_args)  # Convert to list for progress bar tracking
            print("===== All args loaded =====")

            with ProcessPoolExecutor(max_workers=32) as executor:
                futures = []
                for arg in all_args:
                    futures.append(executor.submit(get_data, *arg))

                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()

        else:
            for id in tqdm(os.listdir(args.folder_model)):
                if os.path.isdir(os.path.join(args.folder_model, id)):
                    model_path = os.path.join(args.folder_model, id, f"{id}.obj")
                    smpl_path = os.path.join(args.folder_smpl, id, f"mesh_smpl_{id}.obj")
                    get_data(args=args, model_path=model_path, smpl_path=smpl_path, id=id)

    elif args.mode == 1:
        get_data(args=args, model_path=model_path, smpl_path=smpl_path)

    else:
        assert 1 == 0
        

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parallel', default=True, help='mode to shift', type=bool)
    parser.add_argument('-m', '--mode', default=0, help='mode to shift', type=int)
    parser.add_argument('-l', '--ray_length', default=0.16, type=float)
    parser.add_argument('-l_ops', '--ray_length_ops', default=0.025, type=float)
    # parser.add_argument('-d', '--d_', default=1e-3, type=float)
    parser.add_argument('--num_samples', default=30000, type=int)
    # parser.add_argument('--ply_types', default=["original", "correspondence", "part_label"], type=list)
    # parser.add_argument('-skip', '--skip_handsfeet', default=False, type=bool)


    # mode == 0
    parser.add_argument('-f_m', '--folder_model', default="datafolder/4D-DRESS/data_processed/model", help='Path to the folder containing input scan obj files', type=str)
    parser.add_argument('-f_s', '--folder_smpl', default="datafolder/4D-DRESS/data_processed/smplh", help='Path to the folder containing input smplh obj files', type=str)
    parser.add_argument('-f_p', '--folder_ply', default="datafolder/gt_4D-Dress_data/ply", help='folder containing output ply', type=str)
    parser.add_argument('-f_n', '--folder_npz', default="datafolder/gt_4D-Dress_data/npz", help='folder containing output npz', type=str)

    # mode == 1
    parser.add_argument('-s', '--scan_obj', help='Path to scan obj file', type=str)
    parser.add_argument('-b', '--body_obj', help='Path to body obj file', type=str)

    
    args = parser.parse_args()
    if args.parallel:
        assert args.mode == 0


    # print("\n==== Please make sure you are using env: conver_smpl for trimesh acceleration !!! ====\n")
    main(args)


