from model.split import un_batch
from structs.torch import shape
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

from .spline import BSplineCubic


class SinCos(nn.Module):
  def __init__(self):
    super(SinCos, self).__init__()

  def forward(self, x):
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)



def linear_weights(weight, bias=None, trainable=True):
  linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)

  linear.weight.copy_(weight)
  if bias is not None:
    linear.bias.copy_(bias)

  linear.requires_grad_(trainable)
  return linear

"""
"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
    https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
"""
def random_fourier(input_channels, output_size=256, scale=20):
  basis = linear_weights(
    torch.randn(output_size // 2, input_channels).pow(2) * scale * 2 * np.pi,
    torch.randn(output_size // 2) * 2 * np.pi
  )
  return nn.Sequential(basis, SinCos())


def make_basis(f, input_channels, n = 10):
  directions = [
    [f(i) if c == d else 0 for c in range(input_channels)]
      for i in range(0, n)
        for d in range(0, input_channels)]
  return torch.Tensor(directions)


"""
NeRF fourier positional encoding.
"""

def positional_fourier(input_channels, num_octaves=10):
  basis = linear_weights(
    make_basis(lambda i: 2 ** i, input_channels, num_octaves))
  return nn.Sequential(basis, SinCos())


def random_spline(input_channels, num_splines=64, 
  num_points=64, repeating=True, scale=2):

  basis = linear_weights(
    torch.randn(num_splines, input_channels).pow(2) * scale,
    torch.randn(num_splines) * scale
  )
  return nn.Sequential(basis, 
    BSplineCubic.randn(num_splines, num_points, 1, repeating=repeating))



class FeatureGrids(nn.Module):
  def __init__(self, num_images, grid_size=(16, 16),
               num_features=128, interpolation='bicubic'):
    super(FeatureGrids, self).__init__()

    self.image_ids = num_images
    self.interpolation = interpolation
    self.num_images = num_images
    self.grid_size = grid_size

    self.num_features = num_features

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


class Modulated(nn.Module):
  def __init__(self, feature_grid, encoding):
    super(Modulated, self).__init__()

    self.feature_grid = feature_grid
    self.encoding = encoding

    self.register_buffer("grid_size", torch.Tensor(self.feature_grid.grid_size))

  def forward(self, input):
    _, grid = input

    features = self.feature_grid(input)

    grid = (grid * self.grid_size.view(1, 
      self.grid_size.shape[0], 1).expand(grid.shape))
    fourier = un_batch(self.encoding, grid)

    return features * fourier

class Concatenated(nn.Module):
  def __init__(self, feature_grid, encoding):
    super(Concatenated, self).__init__()

    self.feature_grid = feature_grid
    self.encoding = encoding

    self.register_buffer("grid_size", torch.Tensor(self.feature_grid.grid_size))
    self.num_outputs = feature_grid.num_features + encoding.num_outputs

  def forward(self, input):
    _, grid = input

    features = self.feature_grid(input)

    grid = (grid * self.grid_size.view(1, 
      self.grid_size.shape[0], 1).expand(grid.shape))
    fourier = un_batch(self.encoding, grid)
    
    return torch.cat([features, fourier], dim=1)
