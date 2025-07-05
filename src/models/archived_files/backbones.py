# import torch
# from torch import nn
# from torch.nn import functional as F

# # class ResnetPointnet_confidence(nn.Module):
# #     """PointNet-based encoder network with ResNet blocks.

# #     Args:
# #         out_dim (int): dimension of latent code c
# #         hidden_dim (int): hidden dimension of the network
# #         dim (int): input dimensionality (default 3)
# #     """

# #     def __init__(self, hidden_dim, dim, num_parts, out_dim=1, batchnorm=False, **kwargs):
# #         super().__init__()
# #         self.out_dim = out_dim
# #         self.num_parts = num_parts

# #         self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
# #         self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
# #         self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
# #         self.use_block2 = kwargs.get("use_block2", False)
# #         self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim) if self.use_block2 else None
# #         self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
# #         self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)

# #         self.act = nn.ReLU()

# #         # Add part classifier
# #         final_layer_modules = [
# #             module for module in [
# #                 nn.Conv1d(
# #                     hidden_dim, 128, 1),
# #                 nn.BatchNorm1d(128) if batchnorm else None,
# #                 nn.ReLU(),
# #                 nn.Dropout(0.5),
# #                 nn.Conv1d(128, num_parts, 1)
# #             ] if module is not None
# #         ]
# #         self.final_layers = nn.Sequential(*final_layer_modules)

# #         # Add part-specific predictors
# #         final_layer_modules = [
# #             module for module in [
# #                 nn.Conv1d(
# #                     hidden_dim, 128 * self.num_parts, 1),
# #                 nn.ReLU(),
# #                 nn.Dropout(0.3),
# #                 nn.Conv1d(128 * self.num_parts, self.out_dim * self.num_parts, 1, groups=self.num_parts)
# #             ] if module is not None
# #         ]
# #         self.part_predictors = nn.Sequential(*final_layer_modules)

# #     @staticmethod
# #     def pool(x, dim=-1, keepdim=False):
# #         return x.max(dim=dim, keepdim=keepdim)[0]

# #     def forward(self, p):
# #         B, N, feat_dim = p.shape
# #         net = self.fc_pos(p)
# #         net = self.block_0(net)
# #         pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
# #         net = torch.cat([net, pooled], dim=2)

# #         net = self.block_1(net)
# #         pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
# #         net = torch.cat([net, pooled], dim=2)

# #         net = self.block_3(net)
# #         pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
# #         net = torch.cat([net, pooled], dim=2)

# #         net = self.block_4(net) # shape(B, num_points, hidden_dim)

# #         net = net.permute(0, 2, 1).contiguous() # shape(B, hidden_dim, num_points)

# #         part_labels = self.final_layers(net) # shape(B, num_parts, num_points)
# #         parts_softmax = F.softmax(part_labels, dim=1) # shape(B, num_parts, num_points)

# #         pred = self.part_predictors(net) # shape(B, num_parts * out_dim, num_points)
# #         weighted_pred = pred.view(pred.shape[0], self.out_dim, self.num_parts, -1) * parts_softmax.unsqueeze(1) # shape(B, out_dim, num_parts, num_points)
# #         weighted_pred = weighted_pred.sum(dim=2) # shape(B, out_dim, num_points)

# #         weighted_pred = weighted_pred.permute(0, 2, 1).contiguous() # shape(B, num_points, out_dim)

# #         return part_labels, weighted_pred



# class ResnetPointnet_magnitude(nn.Module):
#     """PointNet-based encoder network with ResNet blocks.

#     Args:
#         out_dim (int): dimension of latent code c
#         hidden_dim (int): hidden dimension of the network
#         dim (int): input dimensionality (default 3)
#     """

#     def __init__(self, out_dim, hidden_dim, dim, **kwargs):
#         super().__init__()
#         self.out_dim = out_dim

#         self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
#         self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
#         self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
#         self.use_block2 = kwargs.get("use_block2", False)
#         self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim) if self.use_block2 else None
#         self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
#         self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
#         self.fc_c = nn.Linear(hidden_dim, out_dim)

#         self.act = nn.ReLU()

#     @staticmethod
#     def pool(x, dim=-1, keepdim=False):
#         return x.max(dim=dim, keepdim=keepdim)[0]

#     def forward(self, p):
#         B, N, feat_dim = p.shape
#         net = self.fc_pos(p)
#         net = self.block_0(net)
#         pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
#         net = torch.cat([net, pooled], dim=2)

#         net = self.block_1(net)
#         pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
#         net = torch.cat([net, pooled], dim=2)

#         net = self.block_3(net)
#         pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
#         net = torch.cat([net, pooled], dim=2)

#         net = self.block_4(net) # shape(B, num_points, hidden_dim)

#         c = self.fc_c(self.act(net)) # shape(B, num_points, out_dim)
#         # c = F.softmax(c.view(-1, self.out_dim), dim=-1)
#         # c = c.view(B, N, self.out_dim)

#         return c

# class SimpleMLP(nn.Module):
#     def __init__(self, input_dim=64, hidden_dim=128, output_dim=1):
#         super(SimpleMLP, self).__init__()
        
#         # 定义一个简单的全连接网络
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),  # 输入层到隐藏层
#             nn.ReLU(),                         # 激活函数
#             nn.Linear(hidden_dim, hidden_dim),  # 隐藏层
#             nn.ReLU(),                         # 激活函数
#             nn.Linear(hidden_dim, output_dim),  # 隐藏层到输出层
#             nn.Softplus()                      # Softplus 确保输出为正数
#         )
    
#     def forward(self, x):
#         B, N, C = x.shape  # 输入形状为 (B, N, C)
#         x = x.view(B * N, C)  # 展平为 (B*N, C)
#         output = self.mlp(x)  # 经过 MLP 计算
#         return output.view(B, N, 1)  # 重新 reshape 为 (B, N, 1)
    

# class ResnetBlockFC(nn.Module):
#     """Fully connected ResNet Block class.

#     Args:
#         size_in (int): input dimension
#         size_out (int): output dimension
#         size_h (int): hidden dimension
#     """

#     def __init__(self, size_in, size_out=None, size_h=None):
#         super().__init__()

#         if size_h is None:
#             size_h = min(size_in, size_out)

#         self.size_in = size_in
#         self.size_h = size_h
#         self.size_out = size_out

#         self.fc_0 = nn.Linear(size_in, size_h)
#         self.fc_1 = nn.Linear(size_h, size_out)
#         self.actvn = nn.ReLU()

#         self.shortcut = nn.Linear(size_in, size_out, bias=False)

#         nn.init.zeros_(self.fc_1.weight)

#     def forward(self, x):
#         net = self.fc_0(self.actvn(x))
#         dx = self.fc_1(self.actvn(net))
#         x_s = self.shortcut(x)

#         return x_s + dx

