import smplx
import numpy as np
import trimesh
import argparse

def export_smplmesh():
    body_model = smplx.create(
                model_path="datafolder/body_models/smpl/neutral/SMPL_NEUTRAL_10pc_rmchumpy.pkl",
                model_type='smpl', 
                gender='neutral',
                # use_face_contour=True,
                # use_compressed=False,
                # num_betas=16
                )

    # compute the SMPL vertices in the world coordinate system
    output = body_model(
        return_verts=True,
    )
    vertices = output.vertices.detach().numpy().squeeze()


    mesh = trimesh.Trimesh(vertices=vertices, faces=body_model.faces, process=False, maintain_order=True)
    return mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='datafolder/useful_data_cape')
    parser.add_argument('--TGT_HEIGHT', type=float, default=1.7)
    parser.add_argument('--TGT_CENTER', type=float, default=0.)
    parser.add_argument('--NORMALIZE', type=bool, default=True)
    parser.add_argument('--EXPAND', type=bool, default=False)

    args = parser.parse_args()

    smpl_mesh = export_smplmesh()

    smpl_mesh.export(f'{args.save_dir}/smpl_mesh_original.obj')
    print(f'Saved smpl_mesh_original.obj to {args.save_dir}')

    if args.NORMALIZE:
        span = smpl_mesh.vertices.max(axis=0) - smpl_mesh.vertices.min(axis=0)
        scale = args.TGT_HEIGHT / span.max()

        center = (smpl_mesh.vertices.max(axis=0) + smpl_mesh.vertices.min(axis=0))/2
        center = args.TGT_CENTER - center
        smpl_mesh.vertices += center
        smpl_mesh.vertices *= scale

    # expand
    if args.EXPAND:
        expand_axis = np.argmin(smpl_mesh.vertices.max(axis=0))
        smpl_mesh.vertices[:, expand_axis] *= 2

    print(f'smpl_mesh.vertices.max(): {smpl_mesh.vertices.max(axis=0)}')
    print(f'smpl_mesh.vertices.min(): {smpl_mesh.vertices.min(axis=0)}')
    smpl_mesh.export(f'{args.save_dir}/smpl_mesh_canonical.obj')
    print(f'Saved smpl_mesh_canonical.obj to {args.save_dir}')
