import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
# from tensorboardX import SummaryWriter

import __init__
from global_info import global_info
from common.train_utils import TrainClock
from utils.extensions.chamfer_dist import ChamferDistance
from common.rotations import matrix_to_unit_quaternion, unit_quaternion_to_matrix
from models.networks import PointAE


from vgtk.loss import CrossEntropyLoss
import vgtk.so3conv.functional as L
import wandb
def bp():
    import pdb;pdb.set_trace()

infos           = global_info()
project_path    = infos.project_path
categories_id   = infos.categories_id
categories      = infos.categories

whole_obj = infos.whole_obj
sym_type  = infos.sym_type

def get_network(cfg, name):
    if name == "pointAE":
        return PointAE(cfg)
    else:
        raise NotImplementedError("Got name '{}'".format(name))

class BaseAgent(object):
    """Base trainer that provides common training behavior.
        All customized trainer should be subclass of this class.
    """
    def __init__(self, cfg):
        self.log_dir = cfg.log_dir
        self.model_dir = cfg.model_dir
        self.clock      = TrainClock()
        self.batch_size = cfg.batch_size
        self.use_wandb  = cfg.use_wandb
        # build network
        self.net = self.build_net(cfg)
        self.cfg = cfg

        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(cfg)

        # set lr scheduler
        self.set_scheduler(cfg)

        # # set tensorboard writer
        # self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'tb/train'))
        # self.val_tb   = SummaryWriter(os.path.join(self.log_dir, 'tb/val'))

        sym_dict = infos.sym_type[self.cfg.category]
        chosen_axis = None
        for key, M in sym_dict.items():
            if M > 20:
                chosen_axis = key
                if 'modelnet' in self.cfg.name_dset:
                    chosen_axis = 'z'
        self.chosen_axis = chosen_axis
        if chosen_axis == 'y':
            self.symmetry_axis = torch.Tensor([0, 1, 0]).view(1, 1, 3).contiguous().cuda()
        elif chosen_axis == 'z':
            self.symmetry_axis = torch.Tensor([0, 0, 1]).view(1, 1, 3).contiguous().cuda()

    @abstractmethod
    def build_net(self, cfg):
        raise NotImplementedError

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterion = nn.MSELoss().cuda()
        if 'completion' in self.cfg.task:
            self.chamfer_dist = ChamferDistance()
        if 'ssl' in self.cfg.task or 'so3' in self.cfg.encoder_type:
            self.classifier = CrossEntropyLoss()
            self.anchors = torch.from_numpy(L.get_anchors(self.cfg.model.kanchor)).cuda()
            with torch.no_grad():
                self.anchor_quat = matrix_to_unit_quaternion(self.anchors)

        if self.cfg.pred_t:
            self.render_loss = torch.nn.L1Loss()
            self.chamfer_dist_2d = ChamferDistance() # newly add

    def _setup_metric(self):
        # regressor + classifier
        anchors = torch.from_numpy(L.get_anchors(self.cfg.model.kanchor)).cuda()
        if self.cfg.model.representation == 'quat':
            out_channel = 4
        elif self.cfg.model.representation == 'ortho6d':
            out_channel = 6
        else:
            raise KeyError("Unrecognized representation of rotation: %s"%self.cfg.model.representation)
        print('---setting up metric!!!')
        self.metric = vgtk.MultiTaskDetectionLoss(anchors, nr=out_channel)

    @abstractmethod
    def collect_loss(self):
        """collect all losses into a dict"""
        raise NotImplementedError

    def set_optimizer(self, cfg):
        """set optimizer used in training"""
        self.base_lr = cfg.lr
        if cfg.batch_size == 1:
            self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=1e-4)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)

    def set_scheduler(self, cfg):
        """set lr scheduler used in training"""
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, cfg.lr_decay)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)

        self.net.cuda()

    def load_ckpt(self, name=None, model_dir=None):
        """load checkpoint from saved checkpoint"""
        if name == 'latest':
            pass
        elif name == 'best':
            pass
        else:
            name = "ckpt_epoch{}".format(name)

        if model_dir is None:
            load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        else:
            load_path = os.path.join(model_dir, "{}.pth".format(name))

        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    @abstractmethod
    def forward(self, data):
        """forward logic for your network"""
        raise NotImplementedError

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        """record and update learning rate"""
        # self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        if not self.optimizer.param_groups[-1]['lr'] < self.base_lr / 5.0:
            self.scheduler.step(self.clock.epoch)

    def record_losses(self, loss_dict, mode='train', infos_dict=None):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        # tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            # tb.add_scalar(k, v, self.clock.step)
            if self.use_wandb:
                wandb.log({f'{mode}/{k}': v, 'step': self.clock.step})
        if infos_dict is not None:
            for k, v in infos_dict.items():
                # tb.add_scalar(k, v, self.clock.step)
                if self.use_wandb:
                    wandb.log({f'{mode}/{k}': v, 'step': self.clock.step})
    def train_func(self, data):
        """one step of training"""
        self.net.train()
        self.is_testing = False

        self.forward(data)
        losses = self.collect_loss()
        infos  = self.collect_info()
        self.update_network(losses)
        self.record_losses(losses, 'train', infos_dict=infos)
        return losses, infos

    def val_func(self, data):
        """one step of validation"""
        self.net.eval()
        self.is_testing = True
        with torch.no_grad():
            self.forward(data)
        losses = self.collect_loss()
        infos  = self.collect_info()
        self.record_losses(losses, 'validation', infos_dict=infos)
        return losses, infos

    def visualize_batch(self, data, tb, **kwargs):
        """write visualization results to tensorboard writer"""
        raise NotImplementedError

    # seems vector is more stable
    def collect_info(self):
        if 'pose' in self.cfg.task:
            return self.infos
        else:
            return None
