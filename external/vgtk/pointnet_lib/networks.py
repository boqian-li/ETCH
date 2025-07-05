import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)

from point_transformer_modules import PointTransformerResBlock, PointTransformerDownBlock, PointTransformerUpBlock, MLP

#
class PointTransformer(nn.Module):
    """
    Output: 128 channels
    """
    def __init__(self, num_channels_R=2, R_dim=3, num_in_channels=3):
        super(PointTransformer, self).__init__()
        cfg = {
            "channel_mult": 4,
            "div": 4,
            "pos_mlp_hidden_dim": 64,
            "attn_mlp_hidden_mult": 4,
            "pre_module": {
                "channel": 16,
                "nsample": 16
            },
            "down_module": {
                "npoint": [256, 64, 32, 16],
                "nsample": [10, 16, 16, 16],
                "attn_channel": [16, 32, 64, 64],
                "attn_num": [2, 2, 2, 2]
            },
            "up_module": {
                "attn_num": [1, 1, 1, 1]
            },
            "heads": {
                "R": [128, num_channels_R * R_dim, None],
                "T": [128, 3, None],
                "N": [128, 3, 'sigmoid'],
                "M": [128, num_channels_R, 'softmax'],
            }
        }
        k = cfg['channel_mult']
        div = cfg["div"]
        pos_mlp_hidden_dim = cfg["pos_mlp_hidden_dim"]
        attn_mlp_hidden_mult = cfg["attn_mlp_hidden_mult"]
        pre_module_channel = cfg["pre_module"]["channel"]
        pre_module_nsample = cfg["pre_module"]["nsample"]
        self.num_in_channels = num_in_channels
        self.num_channels_R  = num_channels_R
        self.pre_module = nn.ModuleList([
            MLP(dim=1, in_channel=num_in_channels, mlp=[pre_module_channel * k] * 2, use_bn=True, skip_last=False),
            PointTransformerResBlock(dim=pre_module_channel * k,
                                     div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                     attn_mlp_hidden_mult=attn_mlp_hidden_mult,
                                     num_neighbors=pre_module_nsample)
        ])
        self.down_module = nn.ModuleList()
        down_cfg = cfg["down_module"]

        last_channel = pre_module_channel
        attn_channel = down_cfg['attn_channel']
        down_sample = down_cfg['nsample']
        for i in range(len(attn_channel)):
            out_channel = attn_channel[i]
            self.down_module.append(PointTransformerDownBlock(npoint=down_cfg['npoint'][i],
                                                              nsample=down_sample[i],
                                                              in_channel=last_channel * k,
                                                              out_channel=out_channel * k,
                                                              num_attn=down_cfg['attn_num'][i],
                                                              div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                                              attn_mlp_hidden_mult=attn_mlp_hidden_mult))
            last_channel = out_channel
        up_channel = attn_channel[::-1] + [pre_module_channel]
        up_sample = down_sample[::-1]
        self.up_module = nn.ModuleList()
        up_cfg = cfg["up_module"]
        up_attn_num = up_cfg['attn_num']
        for i in range(len(attn_channel)):
            self.up_module.append(PointTransformerUpBlock(up_sample[i], up_channel[i] * k, up_channel[i + 1] * k, up_attn_num[i],
                                                          div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                                          attn_mlp_hidden_mult=attn_mlp_hidden_mult))

        self.heads = nn.ModuleDict()
        head_cfg = cfg['heads']
        for key, mlp in head_cfg.items():
            self.heads[key] = MLP(dim=1, in_channel=pre_module_channel * k, mlp=mlp[:-1], use_bn=True, skip_last=True,
                                  last_acti=mlp[-1])

    def forward(self, xyz):  # xyz: [B, 3+X, N]
        # is_input_nan = torch.isnan(xyz).any()
        # print('input tensor is ', is_input_nan)
        xyz_list, points_list = [], []
        if xyz.shape[1] > 3: # with extra feat
            xyz, feat = torch.split(xyz, 3, dim=1)
            if self.num_in_channels > 3:
                points = self.pre_module[0](torch.cat([xyz, feat], dim=1))
            else:
                points = self.pre_module[0](feat)
        else:
            points = self.pre_module[0](xyz)
        points = self.pre_module[1](xyz, points)
        xyz_list.append(xyz)
        points_list.append(points)

        for down in self.down_module:
            xyz, points = down(xyz, points)
            xyz_list.append(xyz)
            points_list.append(points)

        for i, up in enumerate(self.up_module):
            points = up(xyz_list[- (i + 1)], xyz_list[- (i + 2)], points, points_list[- (i + 2)])

        output = {}
        for key, head in self.heads.items():
            output[key] = head(points)
            # is_output_nan = torch.isnan(output[key]).any()
            # print('output tensor is ', is_output_nan)

        return output
def bp():
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    bp()
    model = PointTransformer().cuda()

    input = torch.randn((1, 1024, 3)).cuda()

    output = model(input)
    for key, value in output.items():
        print(key, value.shape)
