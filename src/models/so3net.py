import json

import torch
import torch.nn as nn
import vgtk.so3conv.functional as L

from . import so3conv as M


class EquivBackbone(nn.Module):
    def __init__(self, params, config=None):
        super().__init__()

        self.backbone = nn.ModuleList()
        for block_param in params["backbone"]:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))

        self.na_in = params["na"]
        self.config = config

        self.anchors = torch.from_numpy(L.get_anchors(self.config.model.kanchor)).cuda()

    def forward(self, x):
        sample_idx_lists = []
        if x.shape[-1] > 3:
            x = x.permute(0, 2, 1).contiguous()
        x = M.preprocess_input(x, self.na_in, False)

        for block_i, block in enumerate(self.backbone):
            x, sample_idx_list = block(x)
            sample_idx_lists.append(sample_idx_list)  

        return x, sample_idx_lists


def build_model(
    opt,
    mlps=[[32, 32], [64, 64], [128, 128], [256, 256]],
    out_mlps=[128, 128],
    strides=[2, 2, 2, 2],
    initial_radius_ratio=0.2,
    sampling_ratio=0.8,
    sampling_density=0.5,
    kernel_multiplier=2,
    sigma_ratio=0.5,
    xyz_pooling=None,
    to_file=None,
):
    device = torch.device(f"cuda:{0}")
    input_num = opt.model.input_num
    dropout_rate = opt.model.dropout_rate
    temperature = opt.train_loss.temperature
    so3_pooling = opt.model.flag
    input_radius = opt.model.search_radius

    na = 1 if opt.model.kpconv else opt.model.kanchor

    if input_num > 1024:
        sampling_ratio /= input_num / 1024
        strides[0] = int(2 * (input_num / 1024))
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    print("[MODEL] USING RADIUS AT %f" % input_radius)
    params = {"name": "Invariant SPConv Model", "backbone": [], "na": na}
    dim_in = 1

    n_layer = len(mlps)
    stride_current = 1
    stride_multipliers = [stride_current]
    for i in range(n_layer):
        stride_current *= 2
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]

    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]

    radii = [r * input_radius for r in radius_ratio]

    weighted_sigma = [sigma_ratio * radii[0] ** 2]
    for idx, s in enumerate(strides):
        weighted_sigma.append(weighted_sigma[idx] * s)

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0

            stride_conv = i == 0 or xyz_pooling != "stride"

            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i] ** (1 / sampling_density))

            if i == 0 and j == 0:
                neighbor *= int(input_num / 1024)

            kernel_size = 1
            if j == 0:
                inter_stride = strides[i]
                nidx = i if i == 0 else i + 1

                if stride_conv:
                    neighbor *= 2
                    kernel_size = 1
            else:
                inter_stride = 1
                nidx = i + 1

            block_type = "inter_block" if na != 60 else "separable_block"

            inter_param = {
                "type": block_type,
                "args": {
                    "dim_in": dim_in,
                    "dim_out": dim_out,
                    "kernel_size": kernel_size,
                    "stride": inter_stride,
                    "radius": radii[nidx],
                    "sigma": weighted_sigma[nidx],
                    "n_neighbor": neighbor,
                    "lazy_sample": lazy_sample,
                    "dropout_rate": dropout_rate,
                    "multiplier": kernel_multiplier,
                    "activation": "leaky_relu",
                    "pooling": xyz_pooling,
                    "kanchor": na,
                },
            }
            block_param.append(inter_param)

            dim_in = dim_out

        params["backbone"].append(block_param)

    # representation = opt.model.representation
    # params["outblock"] = {
    #     "dim_in": dim_in,
    #     "mlp": out_mlps,
    #     "fc": [64],
    #     "k": 40,
    #     "kanchor": na,
    #     "pooling": so3_pooling,
    #     "representation": representation,
    #     "temperature": temperature,
    # }

    if to_file is not None:
        with open(to_file, "w") as outfile:
            json.dump(params, outfile)

    model = EquivBackbone(params, config=opt).to(device)
    return model
