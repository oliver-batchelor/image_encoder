from model.split import un_batch
from structs.torch import shape
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math



class FourierEncoding(nn.Module):

  def __init__(self, basis : torch.Tensor, phase=None):
    super(FourierEncoding, self).__init__()
    self.register_buffer("basis", 2 * np.pi * basis)

    if phase == None: 
      phase = basis.new_zeros((1,))
    self.register_buffer("phase", phase)  

  @property
  def num_inputs(self):
    return self.basis.shape[0]

  @property
  def num_outputs(self):
    return self.basis.shape[1] * 2 * len(self.phase)

  def forward(self, x):
    assert x.dim() == 2, f'Expected 2D input (B,C) (got shape {x.shape})'
    assert x.shape[1] == self.num_inputs,\
        f"Expected {self.num_inputs} got shape {x.shape}"

    x = x @ self.basis
    x = x.unsqueeze(2).expand(*x.shape, len(self.phase))
    x = (x + self.phase).view(x.shape[0], -1)

    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


"""
"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
    https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
"""
def random_fourier(input_channels, mapping_size=256, scale=10, num_phases=1):

  directions = torch.randn((input_channels, mapping_size)) * scale
  phases = torch.arange(num_phases) / num_phases

  return FourierEncoding(directions, phases)


def make_basis(f, input_channels, n = 10):
  directions = torch.Tensor([
    [f(i) if c == d else 0 for c in range(input_channels)]
      for i in range(0, n)
        for d in range(0, input_channels)])

  return directions.T

"""
NeRF fourier positional encoding.
"""

def positional_fourier(input_channels, num_octaves=10, num_phases=1):

  directions = make_basis(lambda i: 2 ** i, input_channels, num_octaves)
  phases = torch.arange(num_phases) / num_phases

  return FourierEncoding(directions, phases)



def linear_fourier(input_channels, num_octaves=10, num_phases=1):

  directions = make_basis(lambda i: i, input_channels, num_octaves)
  phases = torch.arange(num_phases) / num_phases

  return FourierEncoding(directions, phases)

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
