# """
# Use pointnet++ to encode point cloud features and decode using part-specific predictors.
# Added global feature layer.
# Base code taken from kaolin library.
# Author: Bharat
# Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
# """
# import torch
# from torch import nn
# from torch.nn import functional as F
# from kaolin.models.PointNet2 import PointNet2SetAbstraction, PointNet2FeaturePropagator, separate_xyz_and_features


# class PointNet2Part(nn.Module):
#     def __init__(self,
#                  in_features=0,
#                  num_classes=2,
#                  num_parts=1,
#                  batchnorm=True,
#                  use_xyz_feature=True,
#                  use_random_ball_query=False):
#         super(PointNet2Part, self).__init__()

#         self.num_parts = num_parts
#         self.num_classes = num_classes
#         self.set_abstractions = nn.ModuleList()
#         self.feature_propagators = nn.ModuleList()

#         self.set_abstractions.append(
#             PointNet2SetAbstraction(
#                 num_points_out=1024,
#                 pointnet_in_features=in_features,
#                 pointnet_layer_dims_list=[
#                     [16, 16, 32],
#                     [32, 32, 64],
#                 ],
#                 radii_list=[0.1, 0.2],
#                 num_samples_list=[16, 32],
#                 batchnorm=batchnorm,
#                 use_xyz_feature=use_xyz_feature,
#                 use_random_ball_query=use_random_ball_query
#             )
#         )

#         self.set_abstractions.append(
#             PointNet2SetAbstraction(
#                 num_points_out=256,
#                 pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
#                 ),
#                 pointnet_layer_dims_list=[
#                     [64, 64, 128],
#                     [64, 96, 128],
#                 ],
#                 radii_list=[0.2, 0.4],
#                 num_samples_list=[16, 32],
#                 batchnorm=batchnorm,
#                 use_xyz_feature=use_xyz_feature,
#                 use_random_ball_query=use_random_ball_query
#             )
#         )

#         self.set_abstractions.append(
#             PointNet2SetAbstraction(
#                 num_points_out=64,
#                 pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
#                 ),
#                 pointnet_layer_dims_list=[
#                     [128, 196, 256],
#                     [128, 196, 256],
#                 ],
#                 radii_list=[0.4, 0.8],
#                 num_samples_list=[16, 32],
#                 batchnorm=batchnorm,
#                 use_xyz_feature=use_xyz_feature,
#                 use_random_ball_query=use_random_ball_query
#             )
#         )

#         self.set_abstractions.append(
#                 (
#                 num_points_out=16,
#                 pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
#                 ),
#                 pointnet_layer_dims_list=[
#                     [256, 256, 512],
#                     [256, 384, 512],
#                 ],
#                 radii_list=[0.8, 1.6],
#                 num_samples_list=[16, 32],
#                 batchnorm=batchnorm,
#                 use_xyz_feature=use_xyz_feature,
#                 use_random_ball_query=use_random_ball_query
#             )
#         )

#         # This kaolin implementation might be different than original paper

#         self.feature_propagators.append(
#             PointNet2FeaturePropagator(
#                 num_features=self.set_abstractions[-2].get_num_features_out(),
#                 num_features_prev=self.set_abstractions[-1].get_num_features_out(),
#                 layer_dims=[512, 512],
#                 batchnorm=batchnorm,
#             )
#         )

#         self.feature_propagators.append(
#             PointNet2FeaturePropagator(
#                 num_features=self.set_abstractions[-3].get_num_features_out(),
#                 num_features_prev=self.feature_propagators[-1].get_num_features_out(
#                 ),
#                 layer_dims=[512, 512],
#                 batchnorm=batchnorm,
#             )
#         )

#         self.feature_propagators.append(
#             PointNet2FeaturePropagator(
#                 num_features=self.set_abstractions[-4].get_num_features_out(),
#                 num_features_prev=self.feature_propagators[-1].get_num_features_out(
#                 ),
#                 layer_dims=[256, 256],
#                 batchnorm=batchnorm,
#             )
#         )

#         self.feature_propagators.append(
#             PointNet2FeaturePropagator(
#                 num_features=in_features,
#                 num_features_prev=self.feature_propagators[-1].get_num_features_out(
#                 ),
#                 layer_dims=[128, 128],
#                 batchnorm=batchnorm,
#             )
#         )

#         self.global_feature = PointNet2SetAbstraction(
#                                 num_points_out=None,
#                                 pointnet_in_features=self.feature_propagators[-1].get_num_features_out(),
#                                 pointnet_layer_dims_list=[
#                                     [32, 32, 32],
#                                 ],
#                                 batchnorm=batchnorm,
#                                 use_xyz_feature=use_xyz_feature,
#                                 use_random_ball_query=use_random_ball_query
#                             )

#         # Add part classifier
#         final_layer_modules = [
#             module for module in [
#                 nn.Conv1d(
#                     self.feature_propagators[-1].get_num_features_out(), 128, 1),
#                 nn.BatchNorm1d(128) if batchnorm else None,
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Conv1d(128, num_parts, 1)
#             ] if module is not None
#         ]
#         self.final_layers = nn.Sequential(*final_layer_modules)

#         # Add part-specific predictors
#         final_layer_modules = [
#             module for module in [
#                 nn.Conv1d(
#                     self.feature_propagators[-1].get_num_features_out(), 128 * self.num_parts, 1),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.Conv1d(128 * self.num_parts, num_classes * self.num_parts, 1, groups=self.num_parts)
#             ] if module is not None
#         ]
#         self.part_predictors = nn.Sequential(*final_layer_modules)

#     def forward(self, points):
#         """
#         Args:
#             points (torch.Tensor): shape = (batch_size, num_points, 3 + in_features)
#                 The points to perform segmentation on.
#         Returns:
#             (torch.Tensor): shape = (batch_size, num_points, num_classes)
#                 The score of each point being in each class.
#                 Note: no softmax or logsoftmax will be applied.
#         """
#         xyz, features = separate_xyz_and_features(points)

#         xyz_list, features_list = [xyz], [features]

#         for module in self.set_abstractions:
#             xyz, features = module(xyz, features)
#             xyz_list.append(xyz)
#             features_list.append(features)

#         target_index = -2
#         for module in self.feature_propagators:
#             features_list[target_index] = module(
#                 xyz_list[target_index],
#                 xyz_list[target_index + 1],
#                 features_list[target_index],
#                 features_list[target_index + 1])

#             target_index -= 1

#         parts = self.final_layers(features_list[0]).contiguous()
#         parts_softmax = F.softmax(parts, dim=1)

#         pred = self.part_predictors(features_list[0]).contiguous()
#         weighted_pred = pred.view(pred.shape[0], self.num_classes, self.num_parts, -1) * parts_softmax.unsqueeze(1)
#         weighted_pred = weighted_pred.sum(dim=2)

#         return {'part_labels': parts, 'correspondences': weighted_pred}