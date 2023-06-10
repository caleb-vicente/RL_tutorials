import torch
import torch.nn as nn
import numpy as np


class FeatureExtractorConvolutional(nn.Module):
    def __init__(self, ch_in, conv1_ch, conv2_ch, conv3_ch, conv4_ch):
        super(FeatureExtractorConvolutional, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, conv1_ch, kernel_size=(4, 4))
        self.conv2 = nn.Conv2d(conv1_ch, conv2_ch, kernel_size=(4, 4))
        self.conv3 = nn.Conv2d(conv2_ch, conv3_ch, kernel_size=(4, 4))
        self.conv4 = nn.Conv2d(conv3_ch, conv4_ch, kernel_size=(4, 4))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.squeeze()
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        return x


class OneHeadAttention(nn.Module):
    def __init__(self, ch_in=1, conv1_ch=16, conv2_ch=20, conv3_ch=24, conv4_ch=30,
                 H=28, W=28, node_size=36, lin_hid=100, out_dim=10, sp_coord_dim=2):
        super(OneHeadAttention, self).__init__()
        self.ch_in = ch_in
        self.conv1_ch = conv1_ch
        self.conv2_ch = conv2_ch
        self.conv3_ch = conv3_ch
        self.conv4_ch = conv4_ch
        self.H = H
        self.W = W
        self.node_size = node_size
        self.lin_hid = lin_hid
        self.out_dim = out_dim
        self.sp_coord_dim = sp_coord_dim
        self.N = int(16 ** 2)  # TODO: check why this part is harcoded
        self.att_map = None

        # TODO: FeatureExtractor must be independendant of the head attention
        self.convolutional_layer = FeatureExtractorConvolutional(self.ch_in, self.conv1_ch, self.conv2_ch,
                                                                 self.conv3_ch, self.conv4_ch)

        self.proj_shape = (self.conv4_ch + self.sp_coord_dim, self.node_size)
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
        x = self.convolutional_layer(x)

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
