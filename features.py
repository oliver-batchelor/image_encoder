from model import with_batches
from structs.torch import shape
import torch
from torch import nn
import numpy as np



class RandomFourier(nn.Module):
    """
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    """

    def __init__(self, input_channels, mapping_size=256, scale=10):
        super(RandomFourier, self).__init__()

        self.input_channels = input_channels
        self.mapping_size = mapping_size

        self.register_buffer("B", torch.randn((input_channels, mapping_size)) * scale)


    def forward(self, x):
        assert x.dim() == 2, f'Expected 2D input (B,C) (got shape {x.shape})'

        channels, _ = x.shape
        assert channels == self.input_channels,\
            f"Expected {self.input_channels} got shape {x.shape}"


        x = 2 * np.pi * (x.T @ self.B)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


