import os
import numpy as np
from colour import Color
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from vgtk.pc import save_ply


red = Color('red')
blue = Color('blue')
white = Color('black')
crange = [np.array(c.rgb) for c in list(red.range_to(blue,1000))]
crange = np.array(crange)
wrrange = [np.array(c.rgb) for c in list(white.range_to(red,1000))]
wrrange = np.array(wrrange)

def clip_to_crange(x, spectrum, xmin, xmax):
    x = x.squeeze()
    cscale = len(spectrum)

    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax

    x = (x - xmin) * cscale / (xmax - xmin)
    x = x.astype(np.int32)
    x = np.clip(x, 0, cscale-1)
    return spectrum[x]


def visualize_one_spheres_np(points, anchors, sphere_path, output_path, vmin=None, vmax=None, radius=0.5):
    '''
    Visualize features defined on a single sphere
    points: np array na x c
    anchors:  np array na x 3
    '''

    if not output_path.endswith('.ply'):
        output_path += '.ply'
    anchorsDense = create_anchors_from_ply(sphere_path)
    # small var: red. large var: blue
    densef = propagation(points, anchors, anchorsDense)
    densef = densef.mean(axis=1)
    densec = 255 * clip_to_crange(densef, crange, vmin, vmax) # na x 3 rgb
    anchor_xyz = radius * anchorsDense
    spheres = np.concatenate((anchor_xyz,densec),axis=1)
    save_ply(output_path, spheres, use_color=True)


def visualize_point_feature_np(points, feat, output_path, vmin=None, vmax=None):
    '''
    Visualize point feature given 1D statistics
    points:  Nx3
    feat: N
    '''
    if not output_path.endswith('.ply'):
        output_path += '.ply'
    color = 255 * clip_to_crange(feat, wrrange, vmin, vmax)
    c_xyz = np.concatenate((xyz,color), axis=1)
    save_ply(output_path, c_xyz, use_color=True)


# import ipdb; ipdb.set_trace()
# from vgtk.zpconv import get_anchors, AnchorProp

# def visualize_all_np(output_path, idx=0, support_points=None,
#                      spherical_points=None, radius=0.005,
#                      vmin=None, vmax=None):
#     '''
#     support_points: [b, 3, m]
#     center_points: [b, 3, n]
#     center_feats: [b, c, n, a]
#     anchors: [a, 3]
#     '''
#     if not output_path.endswith('.ply'):
#         output_path += '.ply'
#     # anchors_dense = get_anchors(242)
#     anchors_dense = 242
#     # small var: red. large var: blue
#     anchor_prop = AnchorProp(spherical_points.anchor, anchors_dense, 0.1).to(spherical_points.xyz.device)
#     feat_dense = anchor_prop(spherical_points).feat #[b, c, n, a2]
#     feat_dense = feat_dense[idx]
#     if vmin is None:
#         vmin = feat_dense.sum(0).mean().item() - 1.5 * feat_dense.sum(0).std().item()
#     if vmax is None:
#         vmax = feat_dense.sum(0).mean().item() + 1.5 * feat_dense.sum(0).std().item()
#     feat_dense = feat_dense.sum(0) #[n, a2]
#     colors = 255. * clip_to_crange(feat_dense.view(-1).detach().cpu().numpy(), crange, vmin, vmax) #[n*a2, 3]

#     anchors_xyz = radius * get_anchors(anchors_dense) #[a2, 3]
#     anchors_xyz = anchors_xyz + spherical_points.xyz[idx].t()[:, None, :].detach().cpu() #[n, a2, 3]
#     anchors_xyz = anchors_xyz.reshape(-1, 3).numpy() #[n*a2, 3]

#     anchors_info = np.concatenate((anchors_xyz, colors), axis=1)

#     sp_xyz = support_points.xyz[idx].t().detach().cpu().numpy() #[m, 3]
#     sp_info = np.concatenate((sp_xyz, sp_xyz*0.+255.), axis=1) #[m, 6]
#     all_info = np.concatenate((sp_info, anchors_info), axis=0)

#     save_ply(output_path, all_info, use_color=True)
