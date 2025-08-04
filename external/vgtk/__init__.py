import sys
TRAIN_PATH = "../"
sys.path.insert(0, TRAIN_PATH)

from models.agent_ae import EquivariantPose
from models.so3conv import *
from models import enc_so3net


def get_agent(cfg):
    if cfg.arch_type == 'ae':
        return EquivariantPose(cfg)
    else:
        raise ValueError

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
