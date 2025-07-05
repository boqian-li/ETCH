
import numpy as np
import torch

import epn_gathering as cuda_gather


# promote input in batch
def promote_input(x, n_dim, device=None):
    x_dim, x_shape = x.ndim, x.shape
    dims = (1,) * (n_dim - x_dim)
    x = x.view(dims+x_shape)
    if device is not None:
        x = x.to(device)
    return x

def promote_input_np(x, n_dim):
    x_dim, x_shape = x.ndim, x.shape
    dims = (1,) * (n_dim-x_dim)
    x = x.reshape(dims+x_shape)
    return x


# gather operators with batch
def batch_gather(x, idx, dim=1):
    x = cuda_gather.gather_points_forward(x, idx.int())
    return x

def batch_zip(x, y, idx):
    raise NotImplementedError('batch zip cuda not implemented')

# TODO: decay step
class LearningRateScheduler():
    def __init__(self, optimizer, init_lr, lr_type, decay_step, decay_rate, **kwargs):
        super(LearningRateScheduler, self).__init__()

        self.counter = 0
        self.init_lr = init_lr
        self.lr = init_lr
        self.lr_type = lr_type
        self.optimizer = optimizer
        self.decay_step = decay_step
        self.schedule_func = self._get_schedule_func(lr_type, decay_rate=decay_rate, **kwargs)

    def _get_schedule_func(self, lr_type, **kwargs):
        return getattr(self, f'_{lr_type}')(**kwargs)

    def step(self):
        self.counter += 1

        if self.counter % self.decay_step == 0:
            lr = self.schedule_func(self.counter // self.decay_step)
            print("[Optimizer] Adjusting learning rate %f ---> %f"%(self.lr, lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.lr = lr


        return self.lr

    def _constant(self, decay_rate):
        return lambda x: self.init_lr

    def _exp_decay(self, decay_rate):
        self.decay_rate = decay_rate
        return lambda x: self.init_lr * decay_rate**x



################## todo ####################


def print_tensor_shape(tensor, tensor_name, msg=None):
    print_msg = ""
    if msg is not None:
        print_msg += msg + '!   '
    print_msg += "%s with Shape:"%tensor_name
    print(print_msg, tensor.shape)


def create_anchors_from_ply(ply_path):
    # Mx3
    pts = load_ply(ply_path)
    norms = np.sqrt(np.sum(pts**2, axis=1))
    pts_selected = pts[np.where(norms>0.5)]
    pts_selected /= np.expand_dims(norms[np.where(norms>0.5)],1)
    return pts_selected


def save_list_as_txt(path, arr, header):
    with open(path, 'w') as f:
        if header is not None:
            f.write(header + '\n')
        for v in arr:
            f.write("%s\n"%(str(v)))


def np_expands(tensor, axis, tiles=None):
    if isinstance(axis, int):
        tensor = np.expand_dims(tensor, axis)
    else:
        for ax in axis:
            tensor = np.expand_dims(tensor, ax)

    if tiles is not None:
        return np.tile(tensor, tiles)
    else:
        return tensor
