import numpy as np
import trimesh
from scipy.spatial import KDTree
import argparse
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle as pkl
import random

def save_data(hit_points, vectors, index_list, file_path):
    hit_points = np.stack(hit_points, axis=0)
    vectors = np.stack(vectors, axis=0)
    index_list = np.array(index_list).astype(np.float64).reshape(-1, 1)

    data = np.concatenate([hit_points, vectors, index_list], axis=-1)
    np.save(file_path, data)

def save_points_with_vector(args, hit_points, vectors, file_path):
    num_points = len(hit_points)
    num_vectors = len(vectors)
    assert num_points == num_vectors
    hit_points = np.stack(hit_points, axis=0)
    vectors = np.stack(vectors, axis=0)

    # ply_types = args.ply_types
    
    # 计算向量终点
    vector_end_points = hit_points - vectors

    with open(file_path, 'w') as f:
        # 写入头部信息
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points * 2}\n")  # 点云和向量的终点
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {num_vectors}\n")  # 向量（作为线段）
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        # 写入点云和向量终点（点云颜色为红色，向量终点颜色为蓝色）
        for p in hit_points:
            f.write(f"{p[0]} {p[1]} {p[2]} 255 0 0\n")  # 红色
        for v in vector_end_points:
            f.write(f"{v[0]} {v[1]} {v[2]} 0 0 255\n")  # 蓝色
        
        # 写入向量（作为线段）
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

def get_data(args, smpl_path, id="abcd"):
    body = trimesh.load_mesh(smpl_path, process=False, maintain_order=True)
    body_vertices = body.vertices


    hit_points = []
    vectors = []
    index_list = []
    
    for i, vertex in enumerate(body_vertices):
        # exclude hands and feet vertices
        if args.skip_handsfeet:
            if args.vertex_2_part[i] in ["left_forearm", "right_forearm", "left_foot", "right_foot"]:
                continue

        normal = body.vertex_normals[i]

        ray_origin = vertex
        ray_direction = normal / np.linalg.norm(normal)  

        distance = 0.05

        hit_points.append(vertex) # + ray_direction * distance)
        vectors.append(ray_direction * distance)
        index_list.append(i)
        
        # # 1. intersection in normal direction
        # locations, index_ray, index_tri = scan.ray.intersects_location(
        #     ray_origins=[ray_origin],
        #     ray_directions=[ray_direction],
        #     multiple_hits=False  
        # )
        # if len(locations) > 0:
        #     nearest_intersection = locations[0]
        #     distance = np.linalg.norm(nearest_intersection - vertex)
        #     if distance < ray_length:
        #     # if True:
        #         # 2. intersection in opposite direction
        #         locations__, index_ray, index_tri = scan.ray.intersects_location(
        #             ray_origins=[ray_origin],
        #             ray_directions=[-ray_direction],
        #             multiple_hits=False  
        #         )

        #         if len(locations__) > 0:
        #             nearest_intersection__ = locations__[0]
        #             distance__ = np.linalg.norm(nearest_intersection__ - vertex)
        #             if distance__ < ray_length_ops:
        #                 hit_points.append(vertex + ray_direction * d_)
        #                 vectors.append(ray_direction * d_)
        #                 index_list.append(i)
        #                 continue

        #         # 4. intersection between smpl parts
        #         locations___, index_ray, index_tri = body.ray.intersects_location(
        #             ray_origins=[ray_origin - ray_direction * 0.005],
        #             ray_directions=[-ray_direction],
        #             multiple_hits=False  
        #         )

        #         if len(locations___) > 0:
        #             nearest_intersection___ = locations___[0]
        #             distance___ = np.linalg.norm(nearest_intersection___ - vertex)
        #             if distance___ < 0.03:
        #                 hit_points.append(vertex + ray_direction * d_)
        #                 vectors.append(ray_direction * d_)
        #                 index_list.append(i)
        #                 continue
                

        #         locations_, index_ray, index_tri = body.ray.intersects_location(
        #             ray_origins=[nearest_intersection],
        #             ray_directions=[-ray_direction],
        #             multiple_hits=False  
        #         )
        #         if not (len(locations_) > 0): 
        #             print(f"skip id == {id}")
        #             continue
        #         if np.linalg.norm(locations_[0] - vertex) < 1e-4:
        #             hit_points.append(nearest_intersection)
        #             vectors.append((nearest_intersection - vertex))
        #             index_list.append(i)
        #             continue
        #         else:
        #             hit_points.append(vertex + ray_direction * d_)
        #             vectors.append(ray_direction * d_)
        #             index_list.append(i)
        #             continue
                

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

        # # 3. 
        # hit_points.append(vertex + ray_direction * d_)
        # vectors.append(ray_direction * d_)
        # index_list.append(i)



    if args.mode == 0:
        ply_path = os.path.join(args.folder_ply, f"hitpts_vectors_{id}.ply")
        npy_path = os.path.join(args.folder_npy, f"hitpts_vectors_{id}.npy")


    save_points_with_vector(args, hit_points, vectors, file_path=ply_path)
    save_data(hit_points, vectors, index_list, file_path=npy_path)

def main(args):
    # print(f"\n ray_length={args.ray_length}, ray_length_ops={args.ray_length_ops}, d_={args.d_} \n")

    if args.mode == 0:
        os.makedirs(args.folder_ply, exist_ok=True)
        os.makedirs(args.folder_npy, exist_ok=True)

        if args.parallel:
            # 取得文件夹中的所有数据文件
            smpl_paths = []
            ids = []
            for id in os.listdir(args.folder_smpl):
                id = id.split('.')[0]
                if os.path.isfile(os.path.join(args.folder_smpl, f"{id}.obj")):
                    smpl_path = os.path.join(args.folder_smpl, f"{id}.obj")

                    smpl_paths.append(smpl_path)
                    ids.append(id)
            
            all_args = zip([args] * len(smpl_paths), smpl_paths, ids)
            all_args = list(all_args)  # 转为列表以便于进度条跟踪
            print("===== All args loaded =====")

            with ProcessPoolExecutor(max_workers=32) as executor:
                futures = []
                for arg in all_args:
                    futures.append(executor.submit(get_data, *arg))

                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()

        # else:
        #     for id in tqdm(os.listdir(args.folder_model)):
        #         if os.path.isdir(os.path.join(args.folder_model, id)):
        #             model_path = os.path.join(args.folder_model, id, f"{id}.obj")
        #             smpl_path = os.path.join(args.folder_smpl, id, f"mesh_smpl_{id}.obj")
        #             get_data(args=args, model_path=model_path, smpl_path=smpl_path, id=id)

    # elif args.mode == 1:
    #     get_data(args=args, model_path=model_path, smpl_path=smpl_path)

    else:
        assert 1 == 0
        

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parallel', default=False, help='mode to shift', type=bool)
    parser.add_argument('-m', '--mode', required=True, help='mode to shift', type=int)
    # parser.add_argument('-l', '--ray_length', default=0.16, type=float)
    # parser.add_argument('-l_ops', '--ray_length_ops', default=0.025, type=float)
    # parser.add_argument('-d', '--d_', default=1e-3, type=float)
    parser.add_argument('-skip', '--skip_handsfeet', default=False, type=bool)
    parser.add_argument('--path_parts', default="data/useful_data/smpl_parts_dense.pkl", type=str)
    # parser.add_argument('--canonical_smpl_mesh_path', default="data/correspondence_data_64/smpl_mesh_scaledtransformed.obj", type=str)
    # parser.add_argument('--ply_types', default=["original", "correspondence", "part_label"], type=list)

    # mode == 0
    # parser.add_argument('-f_m', '--folder_model', default="data/THUman2-1_workspace_smplh/model", help='Path to the folder containing input scan obj files', type=str)
    # parser.add_argument('-f_s', '--folder_smpl', default="data/THUman2-1_workspace_smplh/smplh", help='Path to the folder containing input smplh obj files', type=str)
    parser.add_argument('-f_s', '--folder_smpl', default="data/smpl_synthetic_data/train_mesh", help='Path to the folder containing input smplh obj files', type=str)
    parser.add_argument('-f_p', '--folder_ply', help='folder containing output ply', type=str)
    parser.add_argument('-f_n', '--folder_npy', help='folder containing output npy', type=str)

    # # mode == 1
    # parser.add_argument('-s', '--scan_obj', help='Path to scan obj file', type=str)
    # parser.add_argument('-b', '--body_obj', help='Path to body obj file', type=str)


    args = parser.parse_args()
    if args.parallel:
        assert args.mode == 0

    # args.canonical_smpl_mesh = trimesh.load_mesh(args.canonical_smpl_mesh_path, process=False, maintain_order=True)
    # Load smpl seginfo
    with open(args.seg_info_path, 'rb') as f:
        args.all_seginfo = pkl.load(f)
    

    print("Please make sure you are using env: conver_smpl for trimesh acceleration")
    main(args)


