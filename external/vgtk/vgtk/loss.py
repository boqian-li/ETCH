import math
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.nn.modules.batchnorm import _BatchNorm
from vgtk.zpconv.functional import Gathering, batched_index_select, acos_safe
from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, so3_mean
import vgtk.so3conv as sgtk
def bp():
    import pdb;pdb.set_trace()

# ------------------------------------- loss ------------------------------------
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.metric = torch.nn.CrossEntropyLoss(reduction='none') # TODO

    def forward(self, pred, label):
        _, pred_label = pred.max(1)

        pred_label = pred_label.reshape(-1)
        label_flattened = label.reshape(-1)
        acc = (pred_label == label_flattened).sum().float() / float(label_flattened.shape[0])
        # return torch.zeros(label.shape[0]).cuda(), acc
        return self.metric(pred, label), acc

class AttentionCrossEntropyLoss(torch.nn.Module):
    def __init__(self, loss_type, loss_margin):
        super(AttentionCrossEntropyLoss, self).__init__()
        self.metric = CrossEntropyLoss()
        self.loss_type   = loss_type
        self.loss_margin = loss_margin
        self.iter_counter = 0

    def forward(self, pred, label, wts, rlabel, pretrain_step=2000):
        cls_loss, acc = self.metric(pred, label)

        '''
        rlabel: B or Bx60 -> BxC
        wts: BxA or BxCxA -> BxAxC
        '''
        if wts.ndimension() == 3:
            if wts.shape[1] <= rlabel.shape[1]:
                # BxC
                rlabel = rlabel[:, :wts.shape[1]]
            else:
                rlabel = rlabel.repeat(1,10)[:, :wts.shape[1]]
            # BxAxC
            wts = wts.transpose(1,2)

        r_loss, racc = self.metric(wts, rlabel)

        m = self.loss_margin
        loss_type = self.loss_type

        if loss_type == 'schedule':
            cls_loss_wts = min(float(self.iter_counter) / pretrain_step, 1.0)
            loss = cls_loss_wts * cls_loss + (m + 1.0 - cls_loss_wts) * r_loss
        elif loss_type == 'default':
            loss = cls_loss + m * r_loss
        elif loss_type == 'no_reg':
            loss = cls_loss
        else:
            raise NotImplementedError(f"{loss_type} is not Implemented!")

        if self.training:
            self.iter_counter += 1

        return loss, cls_loss, r_loss, acc, racc

def batched_select_anchor(labels, y, rotation_mapping):
    '''
        (b, c, na_tgt, na_src) x (b, na_tgt)
            -> (b, na_src, c)
            -> (b, na, 3, 3)

    '''
    b, na = labels.shape
    preds_rs = labels.view(-1)[:,None]
    y_rs = y.transpose(1,3).contiguous()
    y_rs = y_rs.view(b*na,na,-1)
    # select into (nb, na, nc) features
    y_select = batched_index_select(y_rs, 1, preds_rs).view(b*na,-1)
    # (nb, na, 3, 3)
    pred_RAnchor = rotation_mapping(y_select).view(b,na,3,3).contiguous()
    return pred_RAnchor

class MultiTaskDetectionLoss(torch.nn.Module):
    def __init__(self, anchors, nr=4, w=10, threshold=1.0, ):
        super(MultiTaskDetectionLoss, self).__init__()
        self.classifier = CrossEntropyLoss()
        self.anchors = anchors
        self.nr = nr
        assert nr == 4 or nr == 6
        self.w = w
        self.threshold = threshold
        self.iter_counter = 0

    def forward(self, wts, label, y, gt_R, R_gt=None):
        ''' setting for alignment regression:
                - label (nb, na):
                    label the targte anchor from the perspective of source anchor na
                - wts (nb, na_tgt, na_src) normalized confidence weights
                - y (nb, nr, na_tgt, na_src) features
                - gt_R (nb, na, 3, 3)
                    relative rotation to regress from the perspective of source anchor na
                    Ra_tgti @ gt_R_i @ Ra_srci.T = R_gt for each i
                - R_gt (nb, 3, 3)
                    ground truth relative rotation: R_gt @ R_tgt = R_src

            setting for canonical regression:
                - label (nb)
                - wts (nb, na) normalized confidence weights
                - y (nb, nr, na) features to be mapped to 3x3 rotation matrices
                - gt_R (nb, na, 3, 3) relative rotation between gtR and each anchor
        '''
        mask = None
        nr = self.nr # 4 or 6
        if isinstance(wts, list):
            b = wts[0].shape[0]
            na = wts[0].shape[1]
        else:
            b = wts.shape[0]
            na = wts.shape[1]
        rotation_mapping = compute_rotation_matrix_from_quaternion if nr == 4 else compute_rotation_matrix_from_ortho6d
        true_R = gt_R[:,29] if R_gt is None else R_gt
        if na == 1:
            # single anchor regression problem
            target_R = true_R
            cls_loss = torch.zeros(1)
            r_acc = torch.zeros(1) + 1
            # Bx6 -> Bx3x3
            pred_R = rotation_mapping(y.view(b,nr))
            l2_loss = torch.pow(pred_R - target_R,2).mean()
            loss = self.w * l2_loss
        elif R_gt is not None and label.ndimension() == 2:
            # Alignment setting
            wts = wts.view(b,na,na)
            cls_loss, r_acc = self.classifier(wts, label)

            # first select the chosen target anchor (nb, na_src)
            confidence, preds = wts.max(1)

            # the followings are [nb, na, 3, 3] predictions of relative rotation
            select_RAnchor = batched_select_anchor(label, y, rotation_mapping)
            pred_RAnchor = batched_select_anchor(preds, y, rotation_mapping)

            # normalize the conrfidence
            confidence = confidence / (1e-6 + torch.sum(confidence,1,keepdim=True))

            # nb, na, 3, 3
            anchors_src = self.anchors[None].expand(b,-1,-1,-1).contiguous()
            pred_Rs = torch.einsum('baij, bajk, balk -> bail', \
                                   anchors_src, pred_RAnchor, self.anchors[preds])

            # pred_Rs_with_label = torch.einsum('baij, bajk, balk -> bail', \
            #                        anchors_src, select_RAnchor, self.anchors[label])

            # averaging rotation: closed form under chordal l2 mean
            pred_R = so3_mean(pred_Rs, confidence)

            # option 1: l2 loss for the prediction at each "tight" anchor pair
            l2_loss = torch.pow(gt_R - select_RAnchor,2).mean()

            # option 2: l2 loss based on the relative prediction with gt label
            # l2_loss = torch.pow(true_R - pred_R_with_label,2).mean() # + torch.pow(gt_R - select_RAnchor,2).mean()

            # loss = self.w * l2_loss
            loss = cls_loss + self.w * l2_loss

        else:
            # single shape Canonical Regression setting
            if isinstance(wts, list):
                cls_loss_list, r_acc_list = [], []
                for wts_item in wts:
                    wts_item = wts_item.view(b,-1)
                    cls_loss, r_acc = self.classifier(wts_item, label.view(-1).contiguous())
                    cls_loss_list.append(cls_loss)
                    r_acc_list.append(r_acc)
                min_loss, min_indices = torch.min(torch.stack(cls_loss_list, dim=0), dim=1) # M, B
                cls_loss = min_loss.mean()
                y_cat    = torch.stack(y, dim=0)
                wts_cat  = torch.stack(wts, dim=0)
                wts      = wts_cat[min_indices, torch.arange(b)]
                y        = y_cat[min_indices, torch.arange(b)]
            else:
                wts = wts.view(b,-1)
                cls_loss, r_acc = self.classifier(wts, label.view(-1).contiguous())
                cls_loss = cls_loss.mean()

            pred_RAnchor = rotation_mapping(y.transpose(1,2).contiguous().view(-1,nr)).view(b,-1,3,3)

            # option 1: only learn to regress the closest anchor
            #
            # pred_ra = batched_index_select(pred_RAnchor, 1, label.long().view(b,-1)).view(b,3,3)
            # target_R = batched_index_select(gt_R, 1, label.long().view(b,-1)).view(b,3,3)
            # l2_loss = torch.pow(pred_ra - target_R,2).mean()
            # loss = cls_loss + self.w * l2_loss

            # option 2: regress nearby anchors within an angular threshold
            gt_bias = angle_from_R(gt_R.view(-1,3,3)).view(b,-1)
            mask = (gt_bias < self.threshold)[:,:,None,None].float()
            l2_loss = torch.pow(gt_R * mask - pred_RAnchor * mask,2).sum()
            loss = cls_loss + self.w * l2_loss

            preds = torch.argmax(wts, 1)
            # The actual prediction is the classified anchor rotation @ regressed rotation
            pred_R = batched_index_select(pred_RAnchor, 1, preds.long().view(b,-1)).view(b,3,3)
            pred_R = torch.matmul(self.anchors[preds], pred_R)

        if self.training:
            self.iter_counter += 1
        return loss, cls_loss, self.w * l2_loss, r_acc, mean_angular_error(pred_R, true_R)

def angle_from_R(R):
    return acos_safe(0.5 * (torch.einsum('bii->b',R) - 1))

def mean_angular_error(pred_R, gt_R):
    R_diff = torch.matmul(pred_R, gt_R.transpose(1,2).float())
    angles = angle_from_R(R_diff)
    return angles#.mean()

def pairwise_distance_matrix(x, y, eps=1e-6):
    M, N = x.size(0), y.size(0)
    x2 = torch.sum(x * x, dim=1, keepdim=True).repeat(1, N)
    y2 = torch.sum(y * y, dim=1, keepdim=True).repeat(1, M)
    dist2 = x2 + torch.t(y2) - 2.0 * torch.matmul(x, torch.t(y))
    dist2 = torch.clamp(dist2, min=eps)
    return torch.sqrt(dist2)


def batch_hard_negative_mining(dist_mat):
    M, N = dist_mat.size(0), dist_mat.size(1)
    assert M == N
    labels = torch.arange(N, device=dist_mat.device).view(N, 1).expand(N, N)
    is_neg = labels.ne(labels.t())
    dist_an, _ = torch.min(torch.reshape(dist_mat[is_neg], (N, -1)), 1, keepdim=False)
    return dist_an



class TripletBatchLoss(nn.Module):
    def __init__(self, opt, anchors, sigma=2e-1, \
                 interpolation='spherical', alpha=0.0):
        '''
            anchors: na x 3 x 3, default anchor rotations
            margin: float, for triplet loss margin value
            sigma: float, sigma for softmax function
            loss: str "none" | "soft" | "hard", for loss mode
            interpolation: str "spherical" | "linear"
        '''
        super(TripletBatchLoss, self).__init__()

        # anchors = sgtk.functinoal.get_anchors()
        self.register_buffer('anchors', anchors)

        self.device = opt.device
        self.loss = opt.train_loss.loss_type
        self.margin = opt.train_loss.margin
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.k_precision = 1

        self.iter_counter = 0

    def forward(self, src, tgt, T, equi_src=None, equi_tgt=None):
        # self._init_buffer(src.shape[0])
        self.gt_idx = torch.arange(src.shape[0], dtype=torch.int32).unsqueeze(1).expand(-1, self.k_precision).contiguous().int().to(self.device)
        if self.alpha > 0 and equi_src is not None and equi_tgt is not None:
            # assert hasattr(self, 'attention_params')
            # return self._forward_attention(src, tgt, T, attention_feats)
            return self._forward_equivariance(src, tgt, equi_src, equi_tgt, T)
        else:
            return self._forward_invariance(src, tgt)

    def _forward_invariance(self, src, tgt):
        '''
            src, tgt: [nb, cdim]
        '''
        # L2 distance function
        dist_func = lambda a,b: (a-b)**2
        bdim = src.size(0)

        # furthest positive

        all_dist = pairwise_distance_matrix(src, tgt)
        furthest_positive = torch.diagonal(all_dist)
        closest_negative = batch_hard_negative_mining(all_dist)

        diff = furthest_positive - closest_negative
        if self.loss == 'hard':
            diff = F.relu(diff + self.margin)
        elif self.loss == 'soft':
            diff = F.softplus(diff, beta=self.margin)
        elif self.loss == 'contrastive':
            diff = furthest_positive + F.relu(self.margin - closest_negative)
        # evaluate accuracy
        _, idx = torch.topk(all_dist, k=self.k_precision, dim=1, largest=False)
        accuracy = torch.sum(idx.int() == self.gt_idx).float() / float(bdim)
        # gather info for debugging
        self.match_idx = idx
        self.all_dist = all_dist
        self.fpos = furthest_positive
        self.cneg = closest_negative

        return diff.mean(), accuracy, furthest_positive.mean(), closest_negative.mean()

    def _forward_equivariance(self, src, tgt, equi_src, equi_tgt, T):

        inv_loss, acc, fp, cn = self._forward_invariance(src, tgt)

        # equi feature: nb, nc, na
        # L2 distance function
        dist_func = lambda a,b: (a-b)**2
        bdim = src.size(0)

        # so3 interpolation
        equi_tgt = self._interpolate(equi_tgt, T, sigma=self.sigma).view(bdim, -1)
        equi_srcR = equi_src.view(bdim, -1)

        # furthest positive
        all_dist = pairwise_distance_matrix(equi_srcR, equi_tgt)
        furthest_positive = torch.diagonal(all_dist)
        closest_negative = batch_hard_negative_mining(all_dist)

        diff = furthest_positive - closest_negative
        if self.loss == 'hard':
            diff = F.relu(diff + self.margin)
        elif self.loss == 'soft':
            diff = F.softplus(diff, beta=self.margin)
        elif self.loss == 'contrastive':
            diff = furthest_positive + F.relu(self.margin - closest_negative)
        # evaluate accuracy
        _, idx = torch.topk(all_dist, k=self.k_precision, dim=1, largest=False)
        accuracy = torch.sum(idx.int() == self.gt_idx).float() / float(bdim)

        inv_info = [inv_loss, acc, fp, cn]
        equi_loss = diff.mean()
        total_loss = inv_loss + self.alpha * equi_loss
        equi_info = [equi_loss, accuracy, furthest_positive.mean(), closest_negative.mean()]

        return total_loss, inv_info, equi_info

    def _forward_attention(self, src, tgt, T, feats):
        '''
            src, tgt: [nb, cdim]
            feats: (src_feat, tgt_feat) [nb, 1, na], normalized attention weights to be aligned
        '''
        # confidence divergence ?
        dist_func = lambda a,b: (a-b)**2

        src_wts = feats[0].squeeze().clamp(min=1e-5)
        tgt_wts = feats[1].squeeze().clamp(min=1e-5)

        inv_loss, acc, fpos, cneg = self._forward_invariance(src, tgt)


        loss_type = self.attention_params['attention_type']
        m = self.attention_params['attention_margin']
        pretrain_step = self.attention_params['attention_pretrain_step']

        #### DEPRECATED
        if src_wts.ndimension() == 3:
            src_wts = src_wts.mean(-1)
            tgt_wts = tgt_wts.mean(-1)

        entropy = -(src_wts * src_wts.log() + tgt_wts * tgt_wts.log())
        entropy_loss = 1e-2 * entropy.sum()

        if loss_type == 'no_reg':
            loss = inv_loss
        else:
            raise NotImplementedError(f"{loss_type} is not Implemented!")

        if self.training:
            self.iter_counter += 1

        return loss, inv_loss, entropy_loss, acc, fpos, cneg


    # knn interpolation of rotated feature
    def _interpolate(self, feature, T, knn=3, sigma=1e-1):
        '''
            :param:
                anchors: [na, 3, 3]
                feature: [nb, cdim, na]
                T: [nb, 4, 4] rigid transformations or [nb, 3, 3]
            :return:
                rotated_feature: [nb, cdim, na]
        '''
        bdim, cdim, adim = feature.shape

        R = T[:,:3,:3]

        # b, na, 3, 3
        r_anchors = torch.einsum('bij,njk->bnik', R.transpose(1,2), self.anchors)

        # b, 1, na, k
        influences, idx = self._rotation_distance(r_anchors, self.anchors, k=knn)
        influences = F.softmax(influences/sigma, 2)[:,None]

        idx = idx.view(-1)
        feat = feature[:,:,idx].reshape(bdim, cdim, adim, knn)

        # b, cdim, na x b, na, k -> b, cdim, na, k
        # feat = sgtk.batched_index_select(feature, 2, idx.reshape(bdim, -1)).reshape(bdim, cdim, adim, knn)
        feat = (feat * influences).sum(-1)

        # spherical gaussian function: e^(lambda*(dot(p,v)-1))
        # see https://mynameismjp.wordpress.com/2016/10/09/sg-series-part-2-spherical-gaussians-101/
        # if self.interpolation == 'spherical':
        #     dists = torch.sum(anchors_tgt*tiled_anchors, dim=3) - 1.0
        #     val, idx = dists.topk(k=knn,dim=2, largest=True)
        # else:
        #     dists = torch.sum((anchors_tgt - tiled_anchors)**2, dim=3)
        #     val, idx = dists.topk(k=knn,dim=2, largest=False)
        return feat


    # b,n,3,3 x m,3,3 -> b,n,k
    def _rotation_distance(self, r0, r1, k=3):
        diff_r = torch.einsum('bnij, mjk->bnmik', r0, r1.transpose(1,2))
        traces = torch.einsum('bnmii->bnm', diff_r)
        return traces.topk(k=k, dim=2)
