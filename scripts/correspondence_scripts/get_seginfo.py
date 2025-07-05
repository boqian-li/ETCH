import pickle as pkl
import os
import trimesh
import numpy as np

def visualize_seginfo(all_seginfo, smpl_mesh_path):
    original_smpl_mesh = trimesh.load_mesh(smpl_mesh_path, process=False, maintain_order=True)
    vertex_colors = np.zeros((original_smpl_mesh.vertices.shape[0], 3), dtype=np.uint8)

    for index in range(len(original_smpl_mesh.vertices)):
        part = all_seginfo["vertex_2_part"][index]
        color = all_seginfo["label_2_color"][all_seginfo["part_2_label"][part]]
        vertex_colors[index] = color
        assert part != 'elsepart'

    # Add colors to mesh and save
    original_smpl_mesh.visual.vertex_colors = vertex_colors
    original_smpl_mesh.export('datafolder/useful_data_cape/original_smpl_with_colors.obj')

def main(path_parts, save_path, smpl_mesh_path):

    # Load part_2_vertex
    with open(path_parts, 'rb') as f:
        part_2_vertex = pkl.load(f, encoding='latin-1')

    part_2_vertex['elsepart'] = []
    print(part_2_vertex.keys())

    ### Get part_2_label
    part_2_label = {}
    # labeled_parts = [part for part in part_2_vertex.keys() if part not in unlabeled_parts]
    # for part in labeled_parts:
    #     part_2_label[part] = labeled_parts.index(part)
    
    # for part in unlabeled_parts:
    #     part_2_label[part] = len(labeled_parts)

    # num_label = len(labeled_parts) + (1 if len(unlabeled_parts) > 0 else 0) 
    for i, part in enumerate(part_2_vertex.keys()):
        part_2_label[part] = i
    
    num_label = len(part_2_label)


    ### Get vertex_2_part
    vertex_2_part = {}
    for k, v in part_2_vertex.items():
        for index in v:
            assert index not in vertex_2_part
            assert k != 'elsepart'
            vertex_2_part[index] = k
    print("len(vertex_2_part): ", len(vertex_2_part))
    print("min(vertex_2_part): ", min(vertex_2_part))
    print("max(vertex_2_part): ", max(vertex_2_part))
    
    # # Get label_2_part
    # label_2_part = {v: k for k, v in part_2_label.items()}
    # print("label_2_part: ", label_2_part)
    
    ### Get label_2_color
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (192, 192, 192),  # Silver
        (128, 0, 128),  # Purple
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Navy Blue
        (255, 165, 0),  # Orange
        (128, 128, 128)  # Gray
    ]
    
    # ensure colors do not repeat
    assert len(colors) == len(set(colors))
    assert len(colors) == num_label

    label_2_color = {}
    for label, color in enumerate(colors):
        label_2_color[label] = color

    # Save All seginfo
    all_seginfo = {}
    all_seginfo["part_2_vertex"] = part_2_vertex
    all_seginfo["part_2_label"] = part_2_label
    all_seginfo["vertex_2_part"] = vertex_2_part
    all_seginfo["label_2_color"] = label_2_color

    print(part_2_label)
    print(label_2_color)


    # visualize
    visualize_seginfo(all_seginfo, smpl_mesh_path)


    with open(save_path, 'wb') as f:
        pkl.dump(all_seginfo, f)







if __name__ == "__main__":
    path_parts = "datafolder/useful_data_cape/my_smpl_parts_dense.pkl"
    save_path = "datafolder/useful_data_cape/my_smpl_seginfo.pkl"
    smpl_mesh_path = 'datafolder/useful_data_cape/smpl_mesh_original.obj'
    # unlabeled_parts = [] # ["left_hand", "right_hand"]
    main(path_parts, save_path, smpl_mesh_path)
