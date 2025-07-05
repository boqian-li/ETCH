import json
import pickle as pkl
import trimesh
import numpy as np

def merge_segments(input_file, output_file, smpl_mesh_path):
    with open(input_file, 'r') as f:
        data = json.load(f)

    mesh = trimesh.load_mesh(smpl_mesh_path, process=False, maintain_order=True)
    # Merge rules
    merged_data = {
        'head': set(data['head'] + data['neck']),
        'left_foot': set(data['leftToeBase'] + data['leftFoot']),
        'left_leg': set(data['leftLeg']),
        'left_upper_leg': set(data['leftUpLeg']),
        'left_hand': set(data['leftHand'] + data['leftHandIndex1']),
        'left_forearm': set(data['leftForeArm']),
        'left_arm': set(data['leftArm']),
        'upper_body': set(data['spine1'] + data['spine2'] + data['spine'] + 
                      data['leftShoulder'] + data['rightShoulder'] + data['hips']),
        'right_foot': set(data['rightToeBase'] + data['rightFoot']),
        'right_leg': set(data['rightLeg']),
        'right_upper_leg': set(data['rightUpLeg']),
        'right_hand': set(data['rightHand'] + data['rightHandIndex1']),
        'right_forearm': set(data['rightForeArm']),
        'right_arm': set(data['rightArm'])
    }

    def resolve_conflicts():
        # Move overlapping vertices from head to upper_body
        merged_data['upper_body'].update(merged_data['head'].intersection(merged_data['upper_body']))
        merged_data['head'].difference_update(merged_data['upper_body'])

        # Move overlapping vertices from left_arm to upper_body
        merged_data['left_arm'].update(merged_data['left_arm'].intersection(merged_data['upper_body']))
        merged_data['upper_body'].difference_update(merged_data['left_arm'])

        # Move overlapping vertices from left_forearm to left_arm
        merged_data['left_arm'].update(merged_data['left_forearm'].intersection(merged_data['left_arm']))
        merged_data['left_forearm'].difference_update(merged_data['left_arm'])

        # Move overlapping vertices from left_hand to left_forearm
        merged_data['left_forearm'].update(merged_data['left_hand'].intersection(merged_data['left_forearm']))
        merged_data['left_hand'].difference_update(merged_data['left_forearm'])

        # Right side
        merged_data['right_arm'].update(merged_data['right_arm'].intersection(merged_data['upper_body']))
        merged_data['upper_body'].difference_update(merged_data['right_arm'])

        merged_data['right_arm'].update(merged_data['right_arm'].intersection(merged_data['right_forearm']))
        merged_data['right_forearm'].difference_update(merged_data['right_arm'])

        merged_data['right_forearm'].update(merged_data['right_hand'].intersection(merged_data['right_forearm']))
        merged_data['right_hand'].difference_update(merged_data['right_forearm'])

        # Leg processing
        # Foot and leg
        merged_data['left_foot'].update(merged_data['left_foot'].intersection(merged_data['left_leg']))
        merged_data['left_leg'].difference_update(merged_data['left_foot'])

        # Leg and upper leg
        merged_data['left_upper_leg'].update(merged_data['left_leg'].intersection(merged_data['left_upper_leg']))
        merged_data['left_leg'].difference_update(merged_data['left_upper_leg'])

        # Upper body and upper leg
        merged_data['upper_body'].update(merged_data['upper_body'].intersection(merged_data['left_upper_leg']))
        merged_data['left_upper_leg'].difference_update(merged_data['upper_body'])

        # Right leg processing
        merged_data['right_foot'].update(merged_data['right_foot'].intersection(merged_data['right_leg']))
        merged_data['right_leg'].difference_update(merged_data['right_foot'])

        merged_data['right_upper_leg'].update(merged_data['right_leg'].intersection(merged_data['right_upper_leg']))
        merged_data['right_leg'].difference_update(merged_data['right_upper_leg'])

        merged_data['upper_body'].update(merged_data['upper_body'].intersection(merged_data['right_upper_leg']))
        merged_data['right_upper_leg'].difference_update(merged_data['upper_body'])

    resolve_conflicts()

    # Check for duplicate vertices
    all_vertices = []
    for key in merged_data.keys():
        all_vertices.extend(list(merged_data[key]))
    assert len(set(all_vertices)) == len(all_vertices) and len(all_vertices) == mesh.vertices.shape[0], "There are duplicate vertices"
    
    # Check for missing vertices
    for i in range(mesh.vertices.shape[0]):
        assert i in all_vertices, f"{i} is not in all_vertices"  

    # convert to list
    merged_data = {k: list(v) for k, v in merged_data.items()}

    # save to pkl
    print(merged_data)
    with open(output_file, 'wb') as f:
        pkl.dump(merged_data, f)

if __name__ == "__main__":
    input_file = 'datafolder/useful_data_cape/smpl_vert_segmentation.json'
    output_file = 'datafolder/useful_data_cape/my_smpl_parts_dense.pkl'
    smpl_mesh_path = 'datafolder/useful_data_cape/smpl_mesh_original.obj'
    merge_segments(input_file, output_file, smpl_mesh_path)
