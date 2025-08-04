import trimesh 
import smplx
import os 
import argparse
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def export_smplmesh(args):
    import argparse
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    assert hasattr(args,'smpl_params_path')
    assert hasattr(args, 'smpl_model_path')
    assert hasattr(args, 'save_path')

    body_model = smplx.create(
                model_path=args.smpl_model_path,
                model_type='smpl',
                gender='neutral')

    # load SMPL parameters in the world coordinate system
    smpl_params = np.load(args.smpl_params_path)
    global_orient = smpl_params['global_orient']
    body_pose = smpl_params['body_pose']
    betas = smpl_params['betas']
    transl = smpl_params['transl']

    # compute the SMPL vertices in the world coordinate system
    output = body_model(
        betas=torch.Tensor(betas).view(1, 10),
        body_pose=torch.Tensor(body_pose).view(1, 23, 3),
        global_orient=torch.Tensor(global_orient).view(1, 1, 3),
        transl=torch.Tensor(transl).view(1, 3),
        return_verts=True
    )
    vertices = output.vertices.detach().numpy().squeeze()

    mesh = trimesh.Trimesh(vertices=vertices, faces=body_model.faces, process=False, maintain_order=True)
    mesh.export(f'{args.save_path}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parallel', default=False, type=bool)
    parser.add_argument('--smpl_model_path', default="body_models/smpl/SMPL_NEUTRAL.pkl", type=str)
    args = parser.parse_args()

    params_dir = "data/HuMMan_data/recon_smpl_params"
    output_dir = "data/HuMMan_data/recon_smpl_meshes"
    os.makedirs(output_dir, exist_ok=True)


    if args.parallel:
        # Get all data files in the folder
        all_args = []

        for name in os.listdir(params_dir):
            assert os.path.isdir(os.path.join(params_dir, name))
            for pose_id in os.listdir(os.path.join(params_dir, name, "smpl_params")):
                if not pose_id.endswith('.npz'):
                    continue
                args_ = dict()
                args_['smpl_params_path'] = os.path.join(params_dir, name, "smpl_params", pose_id)
                args_['save_path'] = os.path.join(output_dir, name, pose_id.replace('npz', 'obj', 1))
                args_['smpl_model_path'] = args.smpl_model_path
                os.makedirs(os.path.dirname(args_['save_path']), exist_ok=True)
                all_args.append(args_)
        
        print("===== All args loaded =====")
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = []
            for args_ in all_args:
                futures.append(executor.submit(export_smplmesh, args_))

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()

    else:
        for name in tqdm(os.listdir(params_dir)):
            assert os.path.isdir(os.path.join(params_dir, name))
            for pose_id in os.listdir(os.path.join(params_dir, name, "smpl_params")):
                assert pose_id.endswith(".npz")
                args.smpl_params_path = os.path.join(params_dir, name, "smpl_params", pose_id)
                args.save_path = os.path.join(output_dir, name, pose_id.replace('npz', 'obj', 1))
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                export_smplmesh(args)

