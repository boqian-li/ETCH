import numpy as np
from math import pi ,sin, cos, sqrt

import torch
import torch.nn as nn
from torch.nn import functional as F
from importlib import import_module

import __init__
from common.debugger import *

class PointAE(nn.Module):
    def __init__(self, cfg):
        super(PointAE, self).__init__()
        self.encoder_type = cfg.encoder_type
        self.decoder_type = cfg.decoder_type

        if 'so3net' in self.encoder_type:
            module = import_module('models')
            param_outfile = None
            self.encoder  = getattr(module, cfg.model.model).build_model_from(cfg, param_outfile)

        if 'pose' in cfg.task:
            self.regressor = RegressorFC(cfg.MODEL.num_channels, bn=False)
            if cfg.pred_nocs:
                self.regressor_nocs = RegressorC1D(list(cfg.nocs_features), cfg.latent_dim)
            if cfg.pred_seg:
                self.classifier_seg = RegressorC1D(list(cfg.seg_features), cfg.latent_dim)
            if cfg.pred_conf:
                self.regressor_confi= RegressorC1D(list(cfg.confi_features), cfg.latent_dim)
            if cfg.pred_mode:
                self.classifier_mode= RegressorC1D(list(cfg.mode_features), cfg.MODEL.num_channels_R)

        # completion
        self.decoder = DecoderFC(eval(cfg.dec_features), cfg.latent_dim, cfg.out_points, cfg.dec_bn)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def regress(self, x):
        return self.regressor(x)

    def forward(self, x):
        z = self.encoder(x)
        return z

    def nb_params(self, net_instance):
        """[This property is used to return the number of trainable parameters for a given layer]
        It is useful for debugging and reproducibility.
        Returns:
            [type] -- [description]
        """
        parameters = list(net_instance.parameters())
        model_parameters = filter(lambda p: p.requires_grad, parameters)
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params


def eval_torch_func(key):
    if key == 'sigmoid':
        return nn.Sigmoid()
    elif key == 'tanh':
        return nn.Tanh()
    elif key == 'softmax':
        return nn.Softmax(1)
    else:
        return NotImplementedError


class RegressorC1D(nn.Module):
    def __init__(self, out_channels=[256, 256, 3], latent_dim=128):
        super(RegressorC1D, self).__init__()
        layers = []#
        out_channels = [latent_dim] + out_channels
        for i in range(1, len(out_channels)-1):
            layers += [nn.Conv1d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.LeakyReLU(inplace=True)] # nn.BatchNorm1d(out_channels[i], eps=0.001)
        if out_channels[-1] in ['sigmoid', 'tanh', 'relu', 'softmax']:
            layers +=[eval_torch_func(out_channels[-1])]
        else:
            layers += [nn.Conv1d(out_channels[-2], out_channels[-1], 1, bias=True)]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

class RegressorFC(nn.Module):
    def __init__(self, latent_dim=128, bn=False):
        super(RegressorFC, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        return x # B, 3, 3


class DecoderFC(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model.append(acti_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = x.view((-1, 3, self.output_pts))
        return x
