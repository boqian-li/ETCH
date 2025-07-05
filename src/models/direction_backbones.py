import numpy as np
import torch
from torch import nn


class BatchLinear(nn.Linear):
    """Helper class for linear layers on order-3 tensors.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Use a bias. Defaults to `True`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        nn.init.xavier_normal_(self.weight, gain=1)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        """Forward pass through layer.

        First unroll batch dimension, then pass
        through dense layer, and finally reshape back to a order-3 tensor.
        Args:
              x (tensor): Inputs of shape `(batch, n, in_features)`.
        Returns:
              tensor: Outputs of shape `(batch, n, out_features)`.
        """
        num_functions, num_inputs = x.shape[0], x.shape[1]
        x = x.view(num_functions * num_inputs, self.in_features)
        out = super().forward(x)
        return out.view(num_functions, num_inputs, self.out_features)


class BatchMLP(nn.Module):
    """Helper class for a simple MLP operating on order-3 tensors.

    Stacks
    several `BatchLinear` modules.
    Args:
        in_features (int): Dimensionality of inputs to MLP.
        out_features (int): Dimensionality of outputs of MLP.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.out_features),
            nn.ReLU(),
            nn.Linear(in_features=self.out_features, out_features=self.out_features),
        )

    def forward(self, x):
        """Forward pass through the network.

        Assumes a batch of tasks as input
        to the network.
        Args:
            x (tensor): Inputs of shape
                `(num_functions, num_points, input_dim)`.
        Returns:
            tensor: Representation of shape
                `(num_functions, num_points, output_dim)`.
        """
        if len(x.shape) > 2:
            num_functions, num_points = x.shape[0], x.shape[1]
            x = x.view(num_functions * num_points, -1)
            rep = self.net(x)
            return rep.view(num_functions, num_points, self.out_features)
        else:
            rep = self.net(x)
            return rep


class DotProdAttention(nn.Module):
    """Simple dot-product attention module.

    Can be used multiple times for
    multi-head attention.
    Args:
        embedding_dim (int): Dimensionality of embedding for keys and queries.
        values_dim (int): Dimensionality of embedding for values.
        linear_transform (bool, optional): Use a linear for all embeddings
            before operation. Defaults to `False`.
    """

    def __init__(self, embedding_dim, values_dim, linear_transform=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.values_dim = values_dim
        self.linear_transform = linear_transform

        if self.linear_transform:
            self.key_transform = BatchLinear(self.embedding_dim, self.embedding_dim, bias=False)
            self.query_transform = BatchLinear(self.embedding_dim, self.embedding_dim, bias=False)
            self.value_transform = BatchLinear(self.values_dim, self.values_dim, bias=False)

    def forward(self, keys, queries, values):
        """Forward pass to implement dot-product attention.

        Assumes that
        everything is in batch mode.
        Args:
            keys (tensor): Keys of shape
                `(num_functions, num_keys, dim_key)`.
            queries (tensor): Queries of shape
                `(num_functions, num_queries, dim_query)`.
            values (tensor): Values of shape
                `(num_functions, num_values, dim_value)`.
        Returns:
            tensor: Output of shape `(num_functions, num_queries, dim_value)`.
        """
        if self.linear_transform:
            keys = self.key_transform(keys)
            queries = self.query_transform(queries)
            values = self.value_transform(values)

        dk = keys.shape[-1]
        attn_logits = torch.bmm(queries, keys.permute(0, 2, 1)) / np.sqrt(dk)
        attn_weights = nn.functional.softmax(attn_logits, dim=-1)
        return torch.bmm(attn_weights, values)


class MultiHeadAttention(nn.Module):
    """Implementation of multi-head attention in a batch way.

    Wraps around the
    dot-product attention module.
    Args:
        embedding_dim (int): Dimensionality of embedding for keys, values,
            queries.
        value_dim (int): Dimensionality of values representation. Is same as
            above.
        num_heads (int): Number of dot-product attention heads in module.
    """

    def __init__(self, embedding_dim, value_dim, num_heads):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.head_size = self.embedding_dim // self.num_heads

        self.key_transform = BatchLinear(self.embedding_dim, self.embedding_dim, bias=False)
        self.query_transform = BatchLinear(self.embedding_dim, self.embedding_dim, bias=False)
        self.value_transform = BatchLinear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attention = DotProdAttention(
            embedding_dim=self.embedding_dim, values_dim=self.embedding_dim, linear_transform=False
        )

        self.head_combine = BatchLinear(self.embedding_dim, self.value_dim)

    def forward(self, keys, queries, values):
        """Forward pass through multi-head attention module.

        Args:
            keys (tensor): Keys of shape
                `(num_functions, num_keys, dim_key)`.
            queries (tensor): Queries of shape
                `(num_functions, num_queries, dim_query)`.
            values (tensor): Values of shape
                `(num_functions, num_values, dim_value)`.
        Returns:
            tensor: Output of shape `(num_functions, num_queries, dim_value)`.
        """
        keys = self.key_transform(keys)
        queries = self.query_transform(queries)
        values = self.value_transform(values)

        keys = self._reshape_objects(keys)
        queries = self._reshape_objects(queries)
        values = self._reshape_objects(values)

        attn = self.attention(keys, queries, values)
        attn = self._concat_head_outputs(attn)
        return self.head_combine(attn)

    def _reshape_objects(self, o):
        num_functions = o.shape[0]
        o = o.view(num_functions, -1, self.num_heads, self.head_size)
        o = o.permute(2, 0, 1, 3).contiguous()
        return o.view(num_functions * self.num_heads, -1, self.head_size)

    def _concat_head_outputs(self, attn):
        num_functions = attn.shape[0] // self.num_heads
        attn = attn.view(self.num_heads, num_functions, -1, self.head_size)
        attn = attn.permute(1, 2, 0, 3).contiguous()
        return attn.view(num_functions, -1, self.num_heads * self.head_size)


class StackedMHSA(nn.Module):
    def __init__(self, embedding_dim, value_dim, num_heads, num_layers):
        super().__init__()
        attention_model_list = [
            MultiHeadAttention(embedding_dim, embedding_dim, num_heads) for _ in range(num_layers - 1)
        ]
        attention_model_list.append(MultiHeadAttention(embedding_dim, value_dim, num_heads))
        self.self_attention_layers = nn.ModuleList(attention_model_list)
        self.num_layers = num_layers

    def forward(self, point_feats):
        """
        Parameters
        ----------
        point_feats: N_item * 60 * feat_dim
        Returns
        -------
        """

        for n, self_attention_layer in enumerate(self.self_attention_layers):
            new_point_feats = self_attention_layer(point_feats, point_feats, point_feats)
            if n != self.num_layers - 1:
                point_feats = point_feats + new_point_feats
            else:
                point_feats = new_point_feats

        return point_feats
