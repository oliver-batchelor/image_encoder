from structs.torch import shape
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math


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

    # r = torch.linspace(0, math.log(scale), steps=mapping_size).exp()

    directions = torch.randn((input_channels, mapping_size)) * scale
    self.register_buffer("B", directions)

  def forward(self, x):
    assert x.dim() == 2, f'Expected 2D input (B,C) (got shape {x.shape})'

    _, channels = x.shape
    assert channels == self.input_channels,\
        f"Expected {self.input_channels} got shape {x.shape}"

    x = 2 * np.pi * (x @ self.B)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class FeatureGrids(nn.Module):
  def __init__(self, num_images, grid_size=(16, 16),
               num_features=128, interpolation='bicubic'):
    super(FeatureGrids, self).__init__()

    self.image_ids = num_images
    self.interpolation = interpolation
    self.num_images = num_images

    features = torch.randn(
        (self.num_images, num_features, *grid_size), dtype=torch.float32)
    self.register_parameter("features", nn.Parameter(features))

  def forward(self, input):
    image_index, grid = input

    grid = grid.permute(0, 2, 1).unsqueeze(2)

    samples = F.grid_sample(self.features[image_index], (grid * 2.0) - 1.0,
                         mode=self.interpolation, padding_mode="border", align_corners=False)

    samples = samples.squeeze(3)
    return samples
