import torch
import torch.nn as nn
import numpy as np
import wandb
import torch.nn.functional as F
from os import makedirs, remove
from os.path import exists, join

import __init__
from models.base import BaseAgent, get_network
from common.pose_utils import mean_angular_error, angle_from_R, rot_diff_degree, rot_diff_rad
from common.rotations import matrix_to_unit_quaternion, rotate
from models.pointnet_lib.pointnet2_modules import farthest_point_sample, gather_operation

import vgtk.so3conv.functional as L
from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, so3_mean
from global_info import global_info

infos           = global_info()
project_path    = infos.project_path
delta_R         = infos.delta_R
delta_T         = infos.delta_T

def bp():
    import pdb;pdb.set_trace()

class EquivariantPose(BaseAgent):
    def __init__(self, cfg):
        super(EquivariantPose, self).__init__(cfg)
        self.flip_axis = cfg.category in ['can']
        print('flip axis', self.flip_axis)

    def build_net(self, cfg):
        net = get_network(cfg, "pointAE")
        print(net)
        if cfg.parallel:
            net = nn.DataParallel(net)
        net = net.cuda()
        return net

    def forward(self, data, verbose=False):
        self.infos    = {}
        self.synchronize(data) #
        self.predict(data)
        self.ssl_rt(data)

    def synchronize(self, data):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        target_keys = ['xyz', 'points', 'C', 'T', 'R_gt', 'R', 'R_label']
        for key in target_keys:
            if key in data:
                data[key] = data[key].to(device)
                if (key == 'xyz' or key == 'points') and self.cfg.use_fps_points:
                    fps_idx = farthest_point_sample(data[key], npoint=1024)
                    new_xyz = gather_operation(data[key].permute(0, 2, 1).contiguous(), fps_idx.int())  # [B, C, S]
                    data[key] = new_xyz.permute(0, 2, 1).contiguous()

    def predict(self, data):
        BS = data['points'].shape[0]
        N  = data['points'].shape[1] # B, N, 3
        CS = self.cfg.MODEL.num_channels_R
        # B, 3, N
        input_pts = data['xyz'].permute(0, 2, 1).contiguous()
        input_pts = input_pts - input_pts.mean(dim=-1, keepdim=True)

        self.latent_vect = self.net.encoder(input_pts)
        self.output_T    = self.latent_vect['T']  # 3, N, no activation
        if 'completion' in self.cfg.task:
            if 'ycb' in self.cfg.task:
                self.output_pts = data['full'].permute(0, 2, 1).to(
                    self.latent_vect['R'].device).float()  # [B, N, 3] -> [B, 3, N]
            elif isinstance(self.latent_vect, dict):
                self.output_pts = self.net.decoder(self.latent_vect['0'])
            else:
                self.output_pts = self.net.decoder(self.latent_vect)

    def ssl_rt(self, data):
        input_pts      = data['xyz']
        shift_dis      = input_pts.mean(dim=1, keepdim=True) if self.cfg.pred_t else 0
        target_pts     = data['points']
        target_T       = data['T'].permute(0, 2, 1).contiguous() # B, 3, N

        BS, N = target_pts.shape[0:2]
        self.threshold = 1.0
        nb, nr, na = self.latent_vect['R'].shape  #
        r_gt        = data['R_gt'].float() # GT R,torch.Size([2, 3, 3])
        rlabel_gt   = data['R_label'].view(-1).contiguous()   # GT R label, torch.Size([2])
        ranchor_gt  = data['R'].float() # GT relative R, torch.Size([2, 60, 3, 3])
        with torch.no_grad():
            ranchor_gt_quat = matrix_to_unit_quaternion(ranchor_gt)

        anchors = self.anchors
        rlabel_pred = self.latent_vect['1']

        rotation_mapping = compute_rotation_matrix_from_quaternion if nr == 4 else compute_rotation_matrix_from_ortho6d
        ranchor_pred = rotation_mapping(self.latent_vect['R'].transpose(1,2).contiguous().view(-1,nr)).view(nb,-1,3,3)
        if 'so3' in self.cfg.encoder_type and na > 1:
            pred_R = torch.matmul(anchors, ranchor_pred) # [60, 3, 3], [nb, 60, 3, 3] --> [nb, 60, 3, 3]
        else:
            pred_R = ranchor_pred

        if self.cfg.pred_t:
            if self.cfg.t_method_type == -1:
                pred_T          = self.output_T.permute(0, 2, 1).contiguous().unsqueeze(-1).contiguous()
                pred_T          = torch.matmul(pred_R, pred_T) # nb, na, 3, 1,
            elif self.cfg.t_method_type == 0:
                pred_T          = self.output_T.permute(0, 2, 1).contiguous().unsqueeze(-1).contiguous()
                if na > 1:
                    pred_T      = torch.matmul(anchors, pred_T) # nb, na, 3, 1,
            else: # type 1, 2, 3, dense voting
                pred_T          = self.output_T.permute(0, 2, 1).contiguous().unsqueeze(-1).contiguous() #  [B, num_heads, 3, A]--> nb, na, 3, 1

        np_out = self.output_pts.shape[-1]
        if self.cfg.r_method_type < 1:  # Q to R first
            if self.cfg.pred_t:
                transformed_pts = torch.matmul(pred_R, self.output_pts.unsqueeze(1).contiguous() - 0.5) + pred_T
                transformed_pts = transformed_pts.permute(0, 1, 3, 2).contiguous() # nb, na, np, 3
                dist1, dist2 = self.chamfer_dist(transformed_pts.view(-1, np_out, 3).contiguous(), (input_pts - shift_dis).unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, N, 3).contiguous(), return_raw=True)
            else:
                transformed_pts = torch.matmul(pred_R, self.output_pts.unsqueeze(1).contiguous() - 0.5).permute(0, 1, 3, 2).contiguous() #
                dist1, dist2 = self.chamfer_dist(transformed_pts.view(-1, np_out, 3).contiguous(), input_pts.unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, N, 3).contiguous(), return_raw=True)

        elif self.cfg.r_method_type == 1: # Q
            qw, qxyz = torch.split(self.latent_vect['R'].permute(0, 2, 1).contiguous(), [1, 3], dim=-1)
            # theta_max= torch.Tensor([1.2]).cuda()
            theta_max= torch.Tensor([36/180 * np.pi]).cuda()
            qw       = torch.cos(theta_max) + (1- torch.cos(theta_max)) * F.sigmoid(qw)
            constrained_quat = torch.cat([qw, qxyz], dim=-1)
            ranchor_pred = rotation_mapping(constrained_quat.view(-1,nr)).view(nb,-1,3,3)
            pred_R = torch.matmul(anchors, ranchor_pred) # [60, 3, 3], [nb, 60, 3, 3] --> [nb, 60, 3, 3]

            constrained_quat_tiled = constrained_quat.unsqueeze(2).contiguous().repeat(1, 1, np_out, 1).contiguous() # nb, na, np, 4
            canon_pts= self.output_pts.permute(0, 2, 1).contiguous() - 0.5 # nb, np, 3
            canon_pts_tiled= canon_pts.unsqueeze(1).contiguous().repeat(1, na, 1, 1).contiguous() # nb, na, np, 3

            transformed_pts = rotate(constrained_quat_tiled, canon_pts_tiled) # nb, na, np, 3
            if self.cfg.pred_t:
                transformed_pts = torch.matmul(anchors, transformed_pts.permute(0, 1, 3, 2).contiguous()) + pred_T
                transformed_pts = transformed_pts.permute(0, 1, 3, 2).contiguous()
                dist1, dist2    = self.chamfer_dist(transformed_pts.view(-1, np_out, 3).contiguous(), (input_pts - shift_dis).unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, N, 3).contiguous(), return_raw=True)
            else:
                transformed_pts = torch.matmul(anchors, transformed_pts.permute(0, 1, 3, 2).contiguous())
                transformed_pts = transformed_pts.permute(0, 1, 3, 2).contiguous()
                dist1, dist2    = self.chamfer_dist(transformed_pts.view(-1, np_out, 3).contiguous(), input_pts.unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, N, 3).contiguous(), return_raw=True)
            self.regu_quat_loss = torch.mean( torch.pow( torch.norm(constrained_quat, dim=-1) - 1, 2))

        if 'partial' in self.cfg.task:
            all_dist = (dist2).mean(-1).view(nb, -1).contiguous()
        else:
            all_dist = (dist1.mean(-1) + dist2.mean(-1)).view(nb, -1).contiguous()

        if na > 1:
            min_loss, min_indices = torch.min(all_dist, dim=-1) # we only allow one mode to be True
            if torch.isnan(ranchor_pred).any() or torch.isnan(ranchor_gt).any():
                self.recon_loss  = torch.Tensor([0]).cuda()#
            else:
                self.recon_loss   = min_loss.mean()
            self.rlabel_pred  = min_indices.detach().clone()
            self.rlabel_pred.requires_grad = False
            self.transformed_pts = transformed_pts[torch.arange(0, BS), min_indices] + shift_dis
            self.r_pred = pred_R[torch.arange(0, BS), min_indices].detach().clone() # correct R by searching
            self.r_pred.requires_grad = False
            if self.cfg.pred_t:
                self.t_pred = pred_T.squeeze(-1)[torch.arange(0, BS), min_indices] + shift_dis.squeeze()
            else:
                self.t_pred = None
        else:
            self.transformed_pts = transformed_pts.squeeze()
            self.r_pred = pred_R.squeeze()
            self.recon_loss   = all_dist.mean()
            if self.cfg.pred_t:
                self.t_pred = pred_T[:, 0, :, :].squeeze() + shift_dis.squeeze()
            else:
                self.t_pred = None

        self.infos["recon"] = self.recon_loss
        if self.cfg.eval:
            print('chamferL1', torch.sqrt(self.recon_loss))

    def eval_func(self, data):
        self.net.eval()
        with torch.no_grad():
            self.forward(data)
        if self.cfg.pre_compute_delta:
            self.pose_err = None
            self.pose_info = {'delta_r': self.r_pred, 'delta_t': self.t_pred}
        else:
            self.eval_so3(data)

    def eval_so3(self, data):
        cfg = self.cfg
        BS = data['points'].shape[0]
        N  = data['points'].shape[1]

        if 'complete' in cfg.name_dset:
            name_id = f'{cfg.exp_num}_modelnet40aligned_{cfg.category}'
        elif 'partial' in cfg.name_dset:
            name_id = f'{cfg.exp_num}_modelnet40new_{cfg.category}'
        else:
            name_id = f'{cfg.exp_num}_{cfg.name_dset}_{cfg.category}'

        delta_file = f'{project_path}/equi-pose/utils/cache/{name_id}.npy'
        if exists(delta_file):
            rt_dict = np.load(delta_file, allow_pickle=True).item()
            delta_R[name_id] = rt_dict['delta_r']
            if 'delta_t' in rt_dict:
                delta_T[name_id] = rt_dict['delta_t'].reshape(1, 3)

        if 'ssl' in cfg.task and cfg.eval:
            if name_id in delta_R:
                self.delta_r = torch.from_numpy(delta_R[name_id]).cuda()
            else:
                print('not found precomputed delta_r!!!')
                self.delta_r= torch.eye(3).reshape((1, 3, 3)).repeat(BS, 1, 1).cuda()

            if name_id in delta_T:
                self.delta_t = torch.from_numpy(delta_T[name_id]).cuda()
            else:
                self.delta_t = torch.zeros(1, 3).cuda()
        else:
            self.delta_r = torch.eye(3).reshape((1, 3, 3)).cuda()
            self.delta_t = torch.zeros(1, 3).cuda()
        self.delta_r = self.delta_r.reshape(1, 3, 3)
        self.delta_t = self.delta_t.reshape(1, 3)

        # if cfg.use_axis or chosen_axis is not None:
        pred_rot    = torch.matmul(self.r_pred, self.delta_r.float().permute(0, 2, 1).contiguous())
        gt_rot      = data['R_gt'].cuda()  # [B, 3, 3]
        rot_err     = rot_diff_degree(pred_rot, gt_rot, chosen_axis=self.chosen_axis, flip_axis=self.flip_axis)
        input_pts  = data['xyz'].permute(0, 2, 1).contiguous().cuda()

        if cfg.pred_t:
            # if
            if self.r_pred.shape[-1] < 3:
                pred_center= self.t_pred.unsqueeze(-1)
            else:
                pred_center= self.t_pred.unsqueeze(-1) - torch.matmul(pred_rot, self.delta_t.unsqueeze(-1)) # ?
            gt_center  = data['T'].cuda() # B, 3 # from original to
            trans_err  = torch.norm(pred_center[:, :, 0] - gt_center[:, 0, :], dim=-1)
        else:
            trans_err = torch.zeros(BS).cuda()
            gt_center  = torch.zeros(BS, 1, 3).cuda()
            pred_center= torch.zeros(BS, 3, 1).cuda()

        scale_err = torch.zeros(BS).cuda()
        scale = torch.ones(BS)
        self.pose_err  = {'rdiff': rot_err, 'tdiff': trans_err, 'sdiff': scale_err}
        self.pose_info = {'r_gt': gt_rot, 't_gt': gt_center.mean(dim=1), 's_gt': scale, 'r_pred': pred_rot, 't_pred': pred_center.mean(dim=-1), 's_pred': scale}
        return

    # seems vector is more stable
    def collect_loss(self):
        loss_dict = {}
        if 'pose' in self.cfg.task:
            if self.cfg.use_objective_T:
                loss_dict["regressionT"]= self.regressionT_loss
            if self.cfg.use_objective_R:
                loss_dict["regressionR"]= self.regressionR_loss
            if self.cfg.use_confidence_R:
                loss_dict['confidence'] = self.regressionCi_loss
            if self.cfg.use_objective_M:
                loss_dict['classifyM'] = 0.1 * self.classifyM_loss
            if self.cfg.use_objective_V:
                loss_dict['consistency'] = self.consistency_loss
        if 'completion' in self.cfg.task:
            loss_dict['recon'] = self.recon_loss
            if self.cfg.use_symmetry_loss:
                loss_dict['chirality'] = self.recon_chirality_loss
            if self.cfg.r_method_type == 1:
                loss_dict['regu_quat'] = 0.1 * self.regu_quat_loss # only for MSE loss

        return loss_dict

    def visualize_batch(self, data, mode, **kwargs):
        num = min(data['points'].shape[0], 12)
        target_pts = data['points'].detach().cpu().numpy() # canonical space

        input_pts = data['xyz'].detach().cpu().numpy()
        ids  = data['id']
        idxs = data['idx']
        BS    = data['points'].shape[0]
        N     = input_pts.shape[1]
        save_path = f'{self.cfg.log_dir}/generation'
        if not exists(save_path):
            print('making directories', save_path)
            makedirs(save_path)

        if self.cfg.pred_nocs:
            nocs_err   = self.nocs_err.cpu().detach().numpy() # GT mode degree error, [B, N]
            output_N   = self.output_N.transpose(-1, -2).cpu().detach().numpy() # B, 3, N --> B, N, 3
            if len(nocs_err.shape) < 3:
                nocs_err = nocs_err[:, :, np.newaxis]

            for j in range(nocs_err.shape[-1]):
                if self.output_N.shape[-1] < target_pts.shape[1]:
                    input_pts  = self.latent_vect['xyz']
                save_arr  = np.concatenate([input_pts, nocs_err[:, :, j:j+1], output_N], axis=-1)
                for k in range(num): # batch
                    save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_{idxs[k]}_{j}.txt' #
                    np.savetxt(save_name, save_arr[k])


        # shape reconstruction
        if 'completion' in self.cfg.task:
            outputs_pts     = self.output_pts[:num].transpose(1, 2).detach().cpu().numpy()
            transformed_pts = self.transformed_pts[:num].detach().cpu().numpy()
            if self.use_wandb and self.clock.step % 500 == 0:
                outputs_pts[0] = outputs_pts[0] + np.array([0, 1, 0]).reshape(1, -1)
                pts = np.concatenate([target_pts[0], outputs_pts[0]], axis=0)
                wandb.log({"input+AE_output": [wandb.Object3D(pts)], 'step': self.clock.step})
                # camera space
                pts = np.concatenate([input_pts[0], transformed_pts[0]], axis=0)
                wandb.log({"camera_space": [wandb.Object3D(pts)], 'step': self.clock.step})

            # canon shape, camera shape, input shape
            for k in range(num): # batch
                save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_input.txt'
                np.savetxt(save_name, np.concatenate( [input_pts[k], 0.1 * np.ones((input_pts[k].shape[0], 1))], axis=1))
                save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_target.txt'
                np.savetxt(save_name, np.concatenate( [target_pts[k], 0.25 * np.ones((target_pts[k].shape[0], 1))], axis=1))
                save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_canon.txt'
                np.savetxt(save_name, np.concatenate( [outputs_pts[k], 0.5 * np.ones((outputs_pts[k].shape[0], 1))], axis=1))
                save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_pred.txt'
                np.savetxt(save_name, np.concatenate( [transformed_pts[k], 0.75 * np.ones((transformed_pts[k].shape[0], 1))], axis=1))
