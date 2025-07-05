import torch
from torch import nn
from torch.nn import functional as F

def vector_loss(a, b, loss_type="l2"):
    bs = a.shape[0]
    if loss_type == "l1":
        loss = F.l1_loss(a.reshape(bs, -1), b.reshape(bs, -1))
    elif loss_type == "l2":
        loss = F.mse_loss(a.reshape(bs, -1), b.reshape(bs, -1))
    else:
        print("loss not implemented")
        exit()
    return loss


def norm_loss(recon_x, x, loss_type="l1"):
    bs = x.shape[0]
    if loss_type == "l1":
        loss = F.l1_loss(recon_x.reshape(bs, -1), x.reshape(bs, -1))
    elif loss_type == "l2":
        loss = F.mse_loss(recon_x.reshape(bs, -1), x.reshape(bs, -1))
    else:
        print("loss not implemented")
        exit()
    return loss


class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super().__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, valid=None):
        face = torch.LongTensor(self.face).cuda()
        coord_out = coord_out.reshape(-1, 6890, 3)
        coord_gt = coord_gt.reshape(-1, 6890, 3)

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss


def marker_loss(verts_pred, verts_gt, markers=None):
    loss = F.mse_loss(verts_pred[:, markers, :], verts_gt[:, markers, :])
    return loss
