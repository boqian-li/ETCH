import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from models.models_pointcloud import GT_network_equiv
from data_utils.GT_dataloader import GTDataset
from torch.utils.data import DataLoader
from utils.GT_utils import save_points_with_vector, save_points_with_color
import pickle as pkl
import torch.nn.functional as F
import trimesh
import json
import matplotlib.pyplot as plt
from models.fit_SMPL import fit_smpl

def obtain_spheres_on_positions(positions, color_mode=0, sphere_radius=0.01):
    spheres = []
    for position in positions:  
        # Create a sphere
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        
        # Move the sphere to the vertex position
        sphere.apply_translation(position)
        
        # Set the sphere color to red
        colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]
        sphere.visual.vertex_colors = colors[color_mode]
        
        # Add the sphere to the list
        spheres.append(sphere)
    
    # Combine all spheres into a single mesh
    combined_spheres = trimesh.util.concatenate(spheres)
    
    return combined_spheres


def place_spheres_on_vertices(mesh, vertex_indices, color_mode=0, sphere_radius=0.01):
    spheres = []
    for index in vertex_indices:
        # Get the vertex position
        vertex_position = mesh.vertices[index]
        
        # Create a sphere
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        
        # Move the sphere to the vertex position
        sphere.apply_translation(vertex_position)
        
        # Set the sphere color to red
        colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]
        sphere.visual.vertex_colors = colors[color_mode]
        
        # Add the sphere to the list
        spheres.append(sphere)
    
    # Combine all spheres into a single mesh
    combined_spheres = trimesh.util.concatenate(spheres)
    
    # Combine the original mesh with the spheres
    final_mesh = trimesh.util.concatenate([mesh, combined_spheres])
    
    return final_mesh

def shuffle_label(labels):
    shuffle_dict = {"0": 75, "1": 0, "2": 70, "3": 22, "4": 12, "5": 56, "6": 10, "7": 18, "8": 4, "9": 67, "10": 61, "11": 64, "12": 53, "13": 73, "14": 62, "15": 66, "16": 33, "17": 78, "18": 54, "19": 72, "20": 11, "21": 30, "22": 40, "23": 28, "24": 9, "25": 65, "26": 5, "27": 39, "28": 31, "29": 35, "30": 45, "31": 44, "32": 16, "33": 42, "34": 34, "35": 7, "36": 49, "37": 82, "38": 19, "39": 83, "40": 25, "41": 47, "42": 13, "43": 24, "44": 3, "45": 17, "46": 38, "47": 8, "48": 68, "49": 6, "50": 55, "51": 36, "52": 77, "53": 85, "54": 43, "55": 50, "56": 46, "57": 84, "58": 15, "59": 69, "60": 27, "61": 41, "62": 58, "63": 26, "64": 48, "65": 76, "66": 57, "67": 32, "68": 81, "69": 59, "70": 63, "71": 79, "72": 37, "73": 29, "74": 1, "75": 52, "76": 21, "77": 2, "78": 23, "79": 80, "80": 74, "81": 20, "82": 60, "83": 71, "84": 14, "85": 51}
    labels = np.asarray([shuffle_dict[str(l)] for l in labels])
    return labels

def default_stuff(seed:int):
    torch.manual_seed(seed+1)
    np.random.seed(seed+10)


def eval(args, model, dataloader):
    model.eval()
    pbar = tqdm(dataloader)
    total_v2v = 0.0
    sample_num = 0

    v2v_score_file = os.path.join(args.output_folder, "v2v_score.txt")
    if os.path.exists(v2v_score_file):
        os.remove(v2v_score_file)

    
    for batch_data in pbar:
        with torch.inference_mode():
            hitpts = batch_data["hitpts"].to(args.device) #shape(B, N, 3)
            gt_vectors = batch_data["vectors"].to(args.device) #shape(B, N, 3)
            gt_confidences = batch_data["confidences"].to(args.device) #shape(B, N, 1)
            gt_labels = batch_data["labels"].to(args.device) #shape(B, N)
            id = batch_data["id"] 
            gender_list = batch_data["gender"]
            B = hitpts.shape[0]

            
            PRED_ITEMS = ["confidence", "direction", "magnitude"] # ["confidence", "direction", "magnitude"]
            results, selected_indexs = model(hitpts, pred_items=PRED_ITEMS, direction_mode="standard_vector") 

            if "confidence" in PRED_ITEMS:
                pred_part_labels = results["part_labels"] # shape(B, K, num_parts)
                _, pred_part_labels = torch.max(pred_part_labels, -1) # shape(B, K)

                # pred_mask = pred_part_labels != (len(args.all_seginfo['label_2_color']) - 1) # shape(B, K)

                pred_confidences = results["confidences"] # shape (B, K, 1)

            
            if "direction" in PRED_ITEMS:
                pred_directions = results["direction"] # shape(B, K, 3)

            if "magnitude" in PRED_ITEMS:
                pred_magnitudes = results["magnitude"] # shape(B, K, 1)

            pred_vectors = pred_directions * pred_magnitudes / args.scale_magnitude # shape(B, K, 3)

            # Save results
            hitpts_k = torch.gather(hitpts, 1, selected_indexs) # shape(B, N, 3) -> shape(B, K, 3)
            gt_vectors_k = torch.gather(gt_vectors, 1, selected_indexs) # shape(B, N, 3) -> shape(B, K, 3)
            gt_labels_k = torch.gather(gt_labels, 1, selected_indexs[:, :, 0]) # shape(B, N) -> shape(B, K)
            gt_confidences_k = torch.gather(gt_confidences, 1, selected_indexs[:, :, 0].unsqueeze(-1)) # shape(B, N, 1) -> shape(B, K, 1)

            
            for j in range(B):
                hitpts_ = hitpts_k[j].clone() # shape(K, 3)
                gt_confidences_ = gt_confidences_k[j].clone() # shape(K, 1)
                gt_labels_ = gt_labels_k[j].clone() # shape(K)
                gt_vectors_ = gt_vectors_k[j].clone() # shape(K, 3)

                pred_vectors_ = pred_vectors[j].clone() # shape(K, 3)
                pred_part_labels_ = pred_part_labels[j].clone() # shape(K)
                pred_confidences_ = pred_confidences[j].clone() # shape(K, 1)
                id_ = id[j]

                # save tightness vectors info
                os.makedirs(os.path.join(args.output_folder, f"{id_}"), exist_ok=True)
                np.savez(os.path.join(args.output_folder, f"{id_}", f"tightness_vectors_info_{id_}.npz"), 
                         hitpts=hitpts_.detach().cpu().numpy(), 
                         pred_vectors=pred_vectors_.detach().cpu().numpy(), 
                         pred_part_labels=pred_part_labels_.detach().cpu().numpy(), 
                         pred_confidences=pred_confidences_.detach().cpu().numpy(), 
                         gt_vectors=gt_vectors_.detach().cpu().numpy(), 
                         gt_labels=gt_labels_.detach().cpu().numpy(), 
                         gt_confidences=gt_confidences_.detach().cpu().numpy())
                
                # save hitpts with pred vectors and gt vectors
                os.makedirs(os.path.join(args.output_folder, f"{id_}"), exist_ok=True)
                save_points_with_vector(hitpts_.detach().cpu().numpy(), pred_vectors_.detach().cpu().numpy(), os.path.join(args.output_folder, f"{id_}", f"hitpts_pred_vectors_{id_}.ply"))
                save_points_with_vector(hitpts_.detach().cpu().numpy(), gt_vectors_.detach().cpu().numpy(), os.path.join(args.output_folder, f"{id_}", f"hitpts_gt_vectors_{id_}.ply"))

                cmap = plt.get_cmap('viridis')
                # save hitpts with gt confidences
                max_ = np.max(gt_confidences_[:, 0].detach().cpu().numpy())
                min_ = np.min(gt_confidences_[:, 0].detach().cpu().numpy())
                gt_confidences_colors = cmap((gt_confidences_[:, 0].detach().cpu().numpy() - min_) / (max_ - min_))[:, :3]
                save_points_with_color(hitpts_.detach().cpu().numpy(), gt_confidences_colors, os.path.join(args.output_folder, f"{id_}", f"hitpts_gt_confidences_{id_}.ply"))
                # save hitpts with pred confidences
                max_ = np.max(pred_confidences_[:, 0].detach().cpu().numpy())
                min_ = np.min(pred_confidences_[:, 0].detach().cpu().numpy())
                pred_confidences_colors = cmap((pred_confidences_[:, 0].detach().cpu().numpy() - min_) / (max_ - min_))[:, :3]
                save_points_with_color(hitpts_.detach().cpu().numpy(), pred_confidences_colors, os.path.join(args.output_folder, f"{id_}", f"hitpts_pred_confidences_{id_}.ply"))
                # save hitpts with gt part labels
                gt_labels_colors = cmap(shuffle_label(gt_labels_.detach().cpu().numpy()) / (len(args.markerset) - 1))[:, :3]
                save_points_with_color(hitpts_.detach().cpu().numpy(), gt_labels_colors, os.path.join(args.output_folder, f"{id_}", f"hitpts_gt_part_labels_{id_}.ply"))
                # save hitpts with pred part labels
                pred_part_labels_colors = cmap(shuffle_label(pred_part_labels_.detach().cpu().numpy()) / (len(args.markerset) - 1))[:, :3]
                save_points_with_color(hitpts_.detach().cpu().numpy(), pred_part_labels_colors, os.path.join(args.output_folder, f"{id_}", f"hitpts_pred_part_labels_{id_}.ply"))

                # save gt inner points with gt confidences
                gt_inner_points_ = hitpts_ - gt_vectors_
                save_points_with_color(gt_inner_points_.detach().cpu().numpy(), gt_confidences_colors, os.path.join(args.output_folder, f"{id_}", f"gt_inner_points_gt_confidences_{id_}.ply"))
                # save gt inner points with gt part labels
                save_points_with_color(gt_inner_points_.detach().cpu().numpy(), gt_labels_colors, os.path.join(args.output_folder, f"{id_}", f"gt_inner_points_gt_part_labels_{id_}.ply"))
                # save pred inner points with pred confidences
                pred_inner_points_ = hitpts_ - pred_vectors_
                save_points_with_color(pred_inner_points_.detach().cpu().numpy(), pred_confidences_colors, os.path.join(args.output_folder, f"{id_}", f"pred_inner_points_pred_confidences_{id_}.ply"))
                # save pred inner points with pred part labels
                save_points_with_color(pred_inner_points_.detach().cpu().numpy(), pred_part_labels_colors, os.path.join(args.output_folder, f"{id_}", f"pred_inner_points_pred_part_labels_{id_}.ply"))

        # fit smpl per batch data
        with torch.inference_mode():
            pred_inner_points = hitpts_k - pred_vectors # shape(B, K, 3)
        
        print(gender_list, set(gender_list))
        if len(set(gender_list)) == 1:
            # the same gender
            gender = gender_list[0]
            final_mesh_list, pred_markers_position, valid_mask, output_smpl_info = fit_smpl(args, pred_inner_points, pred_part_labels, pred_confidences, gender) # input shapes: (B,K,3) (B,K) (B,K,1)
        elif len(set(gender_list)) > 1:
            # different genders
            final_mesh_list = []
            pred_markers_position = []
            valid_mask = []
            output_smpl_info = [[], [], [], [], []]
            for l, gender in enumerate(gender_list):
                final_mesh_list_, pred_markers_position_, valid_mask_, output_smpl_info_ = fit_smpl(args, pred_inner_points[l].unsqueeze(0), pred_part_labels[l].unsqueeze(0), pred_confidences[l].unsqueeze(0), gender) # input shapes: (B,K,3) (B,K) (B,K,1)
                final_mesh_list.append(final_mesh_list_[0])
                pred_markers_position.append(pred_markers_position_[0])
                valid_mask.append(valid_mask_[0])
                for i in range(len(output_smpl_info_)):
                    output_smpl_info[i].append(output_smpl_info_[i][0])

            pred_markers_position = torch.stack(pred_markers_position, dim=0) # shape(B, num_markers, 3)
            valid_mask = torch.stack(valid_mask, dim=0) # shape(B, num_markers)
            tmp_output_smpl_info = []
            for info in output_smpl_info:
                tmp_output_smpl_info.append(np.stack(info, axis=0))
            output_smpl_info = tmp_output_smpl_info
        else:
            raise ValueError(f"Unexpected gender list: {gender_list}")
        
        with torch.inference_mode():
            for j in range(B):
                id_ = id[j]

                #### save gt smpl mesh
                gt_smpl_mesh = trimesh.load_mesh(os.path.join(args.smpl_dir, f"{id_}", f"mesh_smpl_{id_}.obj"), process=False, maintain_order=True)
                gt_smpl_mesh.export(os.path.join(args.output_folder, f"{id_}", f"gt_smpl_mesh_{id_}.obj"))
                gt_smpl_mesh_with_markers = place_spheres_on_vertices(gt_smpl_mesh, list(args.markerset.values()), 0)
                gt_smpl_mesh_with_markers.export(os.path.join(args.output_folder, f"{id_}", f"gt_smpl_mesh_with_markers_{id_}.obj"))

                #### save pred markers
                valid_pred_markers_position_ = pred_markers_position[j][valid_mask[j]] # shape(unsure, 3)
                pred_markers = obtain_spheres_on_positions(valid_pred_markers_position_.detach().cpu().numpy(), 2)
                pred_markers.export(os.path.join(args.output_folder, f"{id_}", f"pred_markers_{id_}.obj"))

                #### save forwarded smpl mesh
                final_smpl_mesh = final_mesh_list[j]
                final_smpl_mesh.export(os.path.join(args.output_folder, f"{id_}", f"forwarded_smpl_mesh_on_pred_{id_}.obj"))
                final_smpl_mesh_with_markers = place_spheres_on_vertices(final_smpl_mesh, list(args.markerset.values()), 1)
                final_smpl_mesh_with_markers.export(os.path.join(args.output_folder, f"{id_}", f"forwarded_smpl_mesh_on_pred_with_markers_{id_}.obj"))

                # compute v2v score:
                gt_smpl_vertices = gt_smpl_mesh.vertices
                final_smpl_vertices = final_smpl_mesh.vertices
                v2v = np.mean(np.linalg.norm(gt_smpl_vertices - final_smpl_vertices, axis=1))
                print(f"{id_} v2v: {v2v}")

                # save output smpl info
                np.savez(os.path.join(args.output_folder, f"{id_}", f"output_smpl_info_{id_}.npz"), 
                         body_pose=output_smpl_info[0][j][:21, :], # shape(21, 3)
                         hand_pose=output_smpl_info[0][j][21:23, :], # shape(2, 3)
                         betas=output_smpl_info[1][j], # shape(10)
                         global_orient=output_smpl_info[2][j], # shape(3)
                         transl=output_smpl_info[3][j], # shape(3)
                         joints=output_smpl_info[4][j]) # shape(45, 3)
                
                # print(output_smpl_info[0][j].shape, output_smpl_info[1][j].shape, output_smpl_info[2][j].shape, output_smpl_info[3][j].shape, output_smpl_info[4][j].shape)

                total_v2v += v2v
                sample_num += 1

                with open(os.path.join(args.output_folder, "v2v_score.txt"), "a") as f:
                    f.write(f"{id_}: {v2v}{'  attention, the valid mask is not full' if int(valid_mask[j].sum().item()) != int(valid_mask[j].shape[0]) else ''}\n")

        
    print(f"average v2v: {total_v2v / sample_num}")
    print(f"total v2v: {total_v2v}")
    print(f"sample num: {sample_num}")
    with open(os.path.join(args.output_folder, "v2v_score.txt"), "a") as f:
        f.write("==========\n")
        f.write(f"average v2v: {total_v2v / sample_num}\n")
        f.write(f"total v2v: {total_v2v}\n")
        f.write(f"sample num: {sample_num}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--activated_ids_path", type=str, default="datafolder/useful_data_4d-dress/val_ids_sampled_ratio10.pkl", help="activated ids") #  # datafolder/useful_data_cape/val_ids.pkl 
    parser.add_argument("--scale_magnitude", type=int, default=10, help="scale for magnitude")
    parser.add_argument('--markerset_path', default="datafolder/useful_data_4d-dress/superset_smpl.json", type=str)
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--infopoints_dir", type=str, default="datafolder/gt_4D-Dress_data/npz", help="data dir path of npy files") # datafolder/gt_CAPE_data/npz # 
    parser.add_argument("--scan_dir", type=str, default="datafolder/4D-DRESS/data_processed/model", help="data dir path of obj files") # datafolder/CAPE_reorganized/cape_release/model_reorganized # 
    parser.add_argument("--smpl_dir", type=str, default="datafolder/4D-DRESS/data_processed/smplh", help="data dir path of smpl files")  # datafolder/CAPE_reorganized/cape_release/smpl_reorganized # 

    parser.add_argument('--model_path', type=str, default=None, help="path of pretrained model")
    parser.add_argument(
        "--batch_size", type=int, default=3, metavar="N", 
    )
    parser.add_argument("--num_point", type=int, default=5000, metavar="N", help="point num sampled from mesh surface")

    parser.add_argument("--EPN_input_radius", type=float, default=0.4)
    parser.add_argument("--EPN_layer_num", type=int, default=2, metavar="N")
    parser.add_argument("--i", type=str, default=None, help="")


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.cuda
    args.device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    default_stuff(args.seed)

    output_folder = os.path.join("./all_experiments/experiments", f"eval_outputs_{args.i}")
    os.makedirs(output_folder, exist_ok=True)
    args.output_folder = output_folder

    # load markerset
    with open(args.markerset_path, 'r') as f:
        args.markerset = json.load(f)


    model = GT_network_equiv(option=args).to(args.device)
    model.load_state_dict(torch.load(args.model_path))


    dataset = GTDataset(args)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=16, prefetch_factor=1)
    
    eval(args, model, eval_loader)
