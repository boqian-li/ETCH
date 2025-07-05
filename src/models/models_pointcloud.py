import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F 
from torch import Tensor
import os

from config.EPN_options import get_default_cfg
from .direction_backbones import BatchMLP
from .pointnet2_utils import PointFeatPropagation
from .so3conv import so3_mean
import trimesh

from .pointtransformer_seg import get_pointtransformer_confidence, get_pointtransformer_magnitude
from .direction_backbones import StackedMHSA
from .so3net import build_model

class GT_network_equiv(nn.Module):
    def __init__(
        self,
        option=None,
    ):
        super().__init__()

        """
        EPN: KPConv+Icosahedron
        """

        self.option = option
        model_setting_file = os.path.join(option.output_folder, "EPN_model_setting_json")
        EPN_cfg = get_default_cfg()
        EPN_cfg.model.search_radius = option.EPN_input_radius

        mlp_layers = [[32, 32], [64, 64], [128, 128], [256, 256]]
        strides_layers = [2, 2, 2, 2]

        # standard_mesh = trimesh.load_mesh("data/correspondence_data_64/smpl_mesh_original.obj", process=False, maintain_order=True)
        # self.standard_normals = torch.tensor(standard_mesh.vertex_normals, dtype=torch.float32, device=option.device, requires_grad=False)
        # print("Standard normals loaded, Shape: ", self.standard_normals.shape)
        self.standard_vector = torch.tensor([0, 0, 1], dtype=torch.float32, requires_grad=False) # device=option.device

        EPN_layer_n = option.EPN_layer_num
        EPN_feat_dim = mlp_layers[EPN_layer_n - 1][0]

        selfatten_feat_dim2 = EPN_feat_dim

        self.encoder = build_model(
            EPN_cfg, mlps=mlp_layers[:EPN_layer_n], strides=strides_layers[:EPN_layer_n], to_file=model_setting_file
        )

        # direction
        self.direction_encoder = StackedMHSA(embedding_dim=selfatten_feat_dim2, value_dim=128, num_heads=8, num_layers=2) # 16, 8
        self.direction_predictor = BatchMLP(in_features=128, out_features=128)
        self.so3_reg = nn.Conv1d(128, 1, 1)

        # magnitude
        pointtransformer_magnitude_cfg = {}
        pointtransformer_magnitude_cfg["c"] = EPN_feat_dim + 3
        pointtransformer_magnitude_cfg["k"] = 1 
        self.magnitude_encoder = get_pointtransformer_magnitude(**pointtransformer_magnitude_cfg)

        # confidence
        pointtransformer_confidence_cfg = {}
        pointtransformer_confidence_cfg["c"] = EPN_feat_dim + 3
        pointtransformer_confidence_cfg["k"] = len(option.markerset) 
        self.confidence_encoder = get_pointtransformer_confidence(**pointtransformer_confidence_cfg)

        print(f"====== Using Total {len(option.markerset)} Markers ======")

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def encode(self, f):
        return self.encoder(f)
    
    def preprocess_data(self, xyz, features):
        B, N, C = features.shape  # Get batch size, number of points, and feature dimensions

        # Flatten batch dimension
        p = xyz.view(-1, 3)  # (B*N, 3)
        x = features.view(-1, C)  # (B*N, C)

        # Calculate batch cumulative tensor
        o = torch.tensor([N * (i + 1) for i in range(B)], dtype=torch.int32).to(p.device)  # (B)

        return p, x, o
    
    def decode_confidence(self, inv_feat, xyz):
        B, point_num, C = inv_feat.shape

        # preprocess
        pxo = self.preprocess_data(xyz, inv_feat)
        part_labels, confidences = self.confidence_encoder(pxo) # shape(B, num_points, num_parts), shape(B, num_points, 1)

        return part_labels, confidences


    def decode_magnitude(self, inv_feat, xyz) -> Tensor:
        B, point_num, C = inv_feat.shape
        pxo = self.preprocess_data(xyz, inv_feat)
        pred_magnitudes = self.magnitude_encoder(pxo) # out: shape(B, point_num, 1)
        
        return pred_magnitudes

    def decode_direction(self, equiv_feat, anchors, initial_vectors):
        B, point_num, feat_dim, na = equiv_feat.shape # shape(B, point_num, C, 60)
        # initial_vectors.shape(B, point_num, 3)

        x = self.direction_encoder(equiv_feat.permute(0, 1, 3, 2).reshape(-1, na, feat_dim).contiguous()) # out: shape(B * point_num, 60, value_dim)
        x = self.direction_predictor(x) # out: shape(B * point_num, 60, value_dim)
        anc_w = self.so3_reg(x.permute(0, 2, 1)) # out: shape(B * point_num, 1, 60)

        # anchors.shape(60, 3, 3)
        pred_direction_matrices = so3_mean(anchors[None, ...].repeat(B * point_num, 1, 1, 1), anc_w.squeeze()) # out: (B * point_num, 3, 3)
        pred_direction_matrices = pred_direction_matrices.reshape(B, point_num, 3, 3)

        initial_vectors = initial_vectors.to(pred_direction_matrices.device)
        pred_directions = torch.matmul(pred_direction_matrices, initial_vectors.unsqueeze(-1)).squeeze(-1) # out: (B, point_num, 3)
        
        return pred_directions


    def soft_aggr_norm(self, feat, part_seg):
        """This is using softmax on the predicted weight to aggregate the feature."""

        B, N, part_num = part_seg.shape
        part_seg = part_seg.permute(0, 2, 1)
        part_feat_list = []
        for part_i in range(part_num):
            feat_weight = part_seg[:, part_i, :]
            feat_weight_normalized = F.normalize(feat_weight, p=1, dim=1)
            weighted_feat = feat_weight_normalized[..., None, None] * feat
            part_i_feat = weighted_feat.sum(1)
            part_feat_list.append(part_i_feat.unsqueeze(1))

        part_feat = torch.cat(part_feat_list, dim=1)

        return part_feat

    def forward(
        self, hitpts, pred_items=["direction", "magnitude"], direction_mode="standard_vector"
    ):
        """
        Input hitpts.shape(B, N, 3)
        pred_items: ["direction", "magnitude", "confidence"]
        direction_mode: ["gt_normal", "pred_normal", "standard_vector"]

        Output vectors.shape(B, N, 3)
        """
        B, N, _ = hitpts.size()
        r, sample_idx_lists = self.encode(hitpts)

        equiv_feat_xyz = r.xyz  # shape(B, 3, K)
        S = equiv_feat_xyz.shape[-1]
        equiv_feat = r.feats.permute(0, 1, 3, 2).reshape(B, -1, S) # shape(B, 64, 60, 1723) -> shape(B, 3840, 1723)
        so3_anchors = r.anchors # shape(60, 3, 3)
        
        _, _, K = equiv_feat_xyz.shape
        
        select = False
        assert select == False
        ## 'select' is a flag for randomly selecting K points from input N points, which was a useful function but deprecated now.
        ## so, we use all points here. K == N in the following code. torch.gather just serves as a placeholder. 

        if select:
            selected_indexs = sample_idx_lists[0][0].detach()[:, :K].type(torch.int64)
            selected_indexs = selected_indexs.unsqueeze(-1).expand(-1, -1, 3)  # Change to (B, K, 3)

            point_equiv_feat = equiv_feat.permute(0, 2, 1).reshape(B, K, -1, 60) # Transpose here
            point_inv_feat = point_equiv_feat.mean(-1) # shape(B, K, C)

        else:
            selected_indexs = torch.arange(0, N).repeat(B, 1).unsqueeze(-1).expand(-1, -1, 3).to(equiv_feat_xyz.device)
            # print(selected_indexs.shape, selected_indexs.max(), selected_indexs.min())
            point_equiv_feat = PointFeatPropagation(
                xyz1=hitpts.permute(0, 2, 1), xyz2=equiv_feat_xyz, points2=equiv_feat
            ).reshape(B, N, -1, 60)
            point_inv_feat = point_equiv_feat.mean(-1) # shape(B, N, C) 
            K = N


        # pred_vectors = pred_directions * pred_magnitudes # out: shape(B, K, 3)
        results = {}

        if "confidence" in pred_items:
            if select:
                part_labels, confidences = self.decode_confidence(point_inv_feat, equiv_feat_xyz.permute(0, 2, 1))
            part_labels, confidences = self.decode_confidence(point_inv_feat, hitpts)
            results["confidences"] = confidences
            results["part_labels"] = part_labels

        if "direction" in pred_items:
            if direction_mode == "gt_normal":
                assert 1==0, "Not implemented"
                # selected_index_list = torch.gather(index_list, 1, selected_indexs[:, :, 0]) # shape(B, K)
                # selected_normals = torch.gather(self.standard_normals.unsqueeze(0).expand(B, -1, -1), 1, selected_index_list.unsqueeze(-1).expand(-1, -1, 3)) # shape(B, K, 3)
                # pred_directions = self.decode_direction(point_equiv_feat, so3_anchors, selected_normals) # out: shape(B, K, 3) #

            elif direction_mode == "standard_vector":
                standard_vector = self.standard_vector.repeat(B, K, 1) # shape (3) -> shape (B, K, 3)
                pred_directions = self.decode_direction(point_equiv_feat, so3_anchors, standard_vector) # out: shape(B, K, 3) #
                
            elif direction_mode == "pred_normal":
                assert 1==0

            results["direction"] = pred_directions

        if "magnitude" in pred_items:
            if select:
                raise ValueError("Not implemented when select = False for self.decode_magnitude")
            pred_magnitudes = self.decode_magnitude(point_inv_feat, hitpts) # out: shape(B, K, 1) #
            results["magnitude"] = pred_magnitudes


        return results, selected_indexs


