import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class OneHeadAttention(nn.Module):
    def __init__(self, input_h=28, input_w=28, n_nodes=252, input_feature_size=30, node_size=36, lin_hid=100, out_dim=10,
                 sp_coord_dim=2, feature_extractor=None):
        super(OneHeadAttention, self).__init__()
        self.H = input_h
        self.W = input_w
        self.node_size = node_size
        self.lin_hid = lin_hid
        self.out_dim = out_dim
        self.sp_coord_dim = sp_coord_dim
        self.N = n_nodes
        self.att_map = None
        self.feature_extractor = feature_extractor
        self.input_feature_size = input_feature_size

        self.proj_shape = (self.input_feature_size + self.sp_coord_dim, self.node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.norm_shape = (self.N, self.node_size)
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

    def forward(self, x):
        n_nodes, _, _, _ = x.shape
        x = self.feature_extractor(x)

        _, _, conv_height, conv_width = x.shape
        xcoords = torch.arange(conv_width).repeat(conv_height, 1).float() / conv_width
        ycoords = torch.arange(conv_height).repeat(conv_width, 1).transpose(1, 0).float() / conv_height
        spatial_coords = torch.stack([xcoords, ycoords], dim=0)
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(n_nodes, 1, 1, 1)
        x = torch.cat([x, spatial_coords], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.flatten(1, 2)

        key = self.k_proj(x)
        key = self.k_norm(key)

        query = self.q_proj(x)
        query = self.q_norm(query)

        value = self.v_proj(x)
        value = self.v_norm(value)
        adjacency = torch.einsum('bfe,bge->bfg', query, key)
        adjacency = adjacency / np.sqrt(self.node_size)
        adjacency = torch.nn.functional.softmax(adjacency, dim=2)
        with torch.no_grad():
            self.att_map = adjacency.clone()
        attention = torch.einsum('bfc,bcd->bfd', adjacency, value)
        attention = self.linear1(attention)
        attention = torch.relu(attention)
        attention = self.norm1(attention)
        attention = attention.max(dim=1)[0]
        y = self.linear2(attention)
        y = torch.nn.functional.log_softmax(y, dim=1)
        return y


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_h=28, input_w=28, n_nodes=49, input_feature_size=30, node_size=36, lin_hid=100, out_dim=5,
                 sp_coord_dim=2, feature_extractor=None, n_heads=3):
        super(MultiHeadAttention, self).__init__()
        self.conv_map = None
        self.att_map = None
        self.H = input_h
        self.W = input_w
        self.node_size = node_size
        self.lin_hid = lin_hid
        self.out_dim = out_dim
        self.sp_coord_dim = sp_coord_dim
        self.N = n_nodes
        self.n_heads = n_heads
        self.input_feature_size = input_feature_size
        self.feature_extractor = feature_extractor
    
        self.proj_shape = (self.input_feature_size + self.sp_coord_dim, self.n_heads * self.node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.k_lin = nn.Linear(self.node_size, self.N)  # B
        self.q_lin = nn.Linear(self.node_size, self.N)
        self.a_lin = nn.Linear(self.N, self.N)

        self.node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

    def forward(self, x):
        N, Cin, H, W = x.shape
        x = self.feature_extractor(x)
        with torch.no_grad():
            self.conv_map = x.clone()  # C
        _, _, cH, cW = x.shape
        xcoords = torch.arange(cW).repeat(cH, 1).float() / cW
        ycoords = torch.arange(cH).repeat(cW, 1).transpose(1, 0).float() / cH
        spatial_coords = torch.stack([xcoords, ycoords], dim=0)
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N, 1, 1, 1)
        x = torch.cat([x, spatial_coords], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.flatten(1, 2)

        key = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        key = self.k_norm(key)

        query = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        query = self.q_norm(query)

        value = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        value = self.v_norm(value)

        # Compatibility function is additive attention (original transformer is dot inner product)
        adjacency = torch.nn.functional.elu(self.q_lin(query) + self.k_lin(key))
        adjacency = self.a_lin(adjacency)
        adjacency = torch.nn.functional.softmax(adjacency, dim=3)

        with torch.no_grad():
            self.att_map = adjacency.clone()  # E
        attention = torch.einsum('bhfc,bhcd->bhfd', adjacency, value)

        # Concat all head attention
        attention = rearrange(attention, 'b head n d -> b n (head d)')

        attention = self.linear1(attention)
        attention = torch.relu(attention)
        attention = self.norm1(attention)
        attention = attention.max(dim=1)[0]
        y = self.linear2(attention)
        y = torch.nn.functional.elu(y)
        return y
