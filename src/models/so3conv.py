import torch
import torch.nn as nn
import torch.nn.functional as F
import vgtk.so3conv as sptk


def preprocess_input(x, na, add_center=True):
    has_normals = x.shape[2] == 6

    if add_center and not has_normals:
        center = x.mean(1, keepdim=True)
        x = torch.cat((center, x), dim=1)[:, :-1]
    xyz = x[:, :, :3]
    return sptk.SphericalPointCloud(
        xyz.permute(0, 2, 1).contiguous(), sptk.get_occupancy_features(x, na, add_center), None
    )


class IntraSO3ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, norm=None, activation="relu", dropout_rate=0):
        super().__init__()

        if norm is None:
            norm = nn.InstanceNorm2d

        self.conv = sptk.IntraSO3Conv(dim_in, dim_out)
        self.norm = norm(dim_out, affine=False)

        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.conv(x)
        feat = self.norm(x.feats)
        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)

        return sptk.SphericalPointCloud(x.xyz, feat, x.anchors)


class InterSO3ConvBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride,
        radius,
        sigma,
        n_neighbor,
        multiplier,
        kanchor=60,
        lazy_sample=None,
        norm=None,
        activation="relu",
        pooling="none",
        dropout_rate=0,
    ):
        super().__init__()

        if lazy_sample is None:
            lazy_sample = True
        if norm is None:
            norm = nn.InstanceNorm2d

        pooling_method = None if pooling == "none" else pooling
        self.conv = sptk.InterSO3Conv(
            dim_in,
            dim_out,
            kernel_size,
            stride,
            radius,
            sigma,
            n_neighbor,
            kanchor=kanchor,
            lazy_sample=lazy_sample,
            pooling=pooling_method,
        )
        self.norm = norm(dim_out, affine=False)
        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x, inter_idx=None, inter_w=None):
        inter_idx, inter_w, sample_idx, x = self.conv(x, inter_idx, inter_w)

        feat = self.norm(x.feats)

        if self.relu is not None:
            feat = self.relu(feat)

        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return inter_idx, inter_w, sample_idx, sptk.SphericalPointCloud(x.xyz, feat, x.anchors)


class BasicSO3ConvBlock(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.layer_types = []
        for param in params:
            if param["type"] == "intra_block":
                conv = IntraSO3ConvBlock(**param["args"])
            elif param["type"] == "inter_block":
                conv = InterSO3ConvBlock(**param["args"])
            elif param["type"] == "separable_block":
                conv = SeparableSO3ConvBlock(param["args"])
            else:
                raise ValueError(f'No such type of SO3Conv {param["type"]}')
            self.layer_types.append(param["type"])
            self.blocks.append(conv)
        self.params = params

    def forward(self, x):
        inter_idx, inter_w = None, None
        sample_idx_list = []
        for conv, param in zip(self.blocks, self.params):
            if param["type"] in ["inter", "inter_block", "separable_block"]:
                inter_idx, inter_w, sample_idx, x = conv(x, inter_idx, inter_w)

                if param["args"]["stride"] > 1:
                    inter_idx, inter_w = None, None
            elif param["type"] in ["intra_block"]:
                x = conv(x)
                sample_idx = None
            else:
                raise ValueError(f'No such type of SO3Conv {param["type"]}')
            
            sample_idx_list.append(sample_idx)

        return x, sample_idx_list


class SeparableSO3ConvBlock(nn.Module):
    def __init__(self, params):
        super().__init__()

        dim_in = params["dim_in"]
        dim_out = params["dim_out"]

        self.use_intra = params["kanchor"] > 1

        self.inter_conv = InterSO3ConvBlock(**params)

        intra_args = {
            "dim_in": dim_out,
            "dim_out": dim_out,
            "dropout_rate": params["dropout_rate"],
            "activation": params["activation"],
        }

        if self.use_intra:
            self.intra_conv = IntraSO3ConvBlock(**intra_args)
        self.stride = params["stride"]

        self.skip_conv = nn.Conv2d(dim_in, dim_out, 1)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False)
        self.relu = getattr(F, params["activation"])

    def forward(self, x, inter_idx, inter_w):
        """Inter, intra conv with skip connection."""
        skip_feature = x.feats
        inter_idx, inter_w, sample_idx, x = self.inter_conv(x, inter_idx, inter_w)

        if self.use_intra:
            x = self.intra_conv(x)
        if self.stride > 1:
            skip_feature = sptk.functional.batched_index_select(skip_feature, 2, sample_idx.long())
        skip_feature = self.skip_conv(skip_feature)
        skip_feature = self.relu(self.norm(skip_feature))
        x_out = sptk.SphericalPointCloud(x.xyz, x.feats + skip_feature, x.anchors)
        return inter_idx, inter_w, sample_idx, x_out


def so3_mean(Rs, weights=None):
    """Get the mean of the rotations.
    Parameters
    ----------
    Rs: (B,N,3,3)
    weights : array_like shape (B,N,), optional
        Weights describing the relative importance of the rotations. If
        None (default), then all values in `weights` are assumed to be
        equal.
    Returns
    -------
    mean R: (B,3,3)
    -----
    The mean used is the chordal L2 mean (also called the projected or
    induced arithmetic mean). If ``p`` is a set of rotations with mean
    ``m``, then ``m`` is the rotation which minimizes
    ``(weights[:, None, None] * (p.as_matrix() - m.as_matrix())**2).sum()``.
    """
    nb, na, _, _ = Rs.shape
    mask = torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]).float().to(Rs.device)
    mask2 = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]]).float().to(Rs.device)
    mask = mask[None].expand(nb, -1, -1).contiguous()
    mask2 = mask2[None].expand(nb, -1, -1).contiguous()

    if weights is None:
        weights = 1.0
    else:
        weights = weights[:, :, None, None]

    Ce = torch.sum(weights * Rs, dim=1)
    try:
        cu, cd, cv = torch.svd(Ce)
    except Exception:
        cu, cd, cv = torch.svd(Ce + 1e-4 * Ce.mean() * torch.rand_like(Ce))

    cvT = cv.transpose(1, 2).contiguous()
    dets = torch.det(torch.matmul(cu, cvT))

    D = mask * dets[:, None, None] + mask2
    return torch.einsum("bij,bjk,bkl->bil", cu, D, cvT)
