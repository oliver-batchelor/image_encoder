from structs.torch import shape
from torch import nn
import torch
from torch.nn import functional as F
import math


class Sine(nn.Module):
  def __init__(self, w0=1.):
    super().__init__()
    self.w0 = w0

  def forward(self, x):
    return torch.sin(self.w0 * x)

# siren layer


class Siren(nn.Module):
  def __init__(self, input_size, output_size, w_std, activation=None):
    super().__init__()
    self.dim_in = input_size

    self.linear = nn.Linear(input_size, output_size)
    self.activation = activation or nn.Identity()

    self.linear.weight.uniform_(-w_std, w_std)
    if self.linear.bias is not None:
      self.linear.bias.uniform_(-w_std, w_std)



  def forward(self, x):
    return self.activation(self.linear(x))
   


class SirenNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size=3, 
      hidden_layers=1, w0=1, w0_initial=30, c=6):
    super().__init__()

    w_std = math.sqrt(c / hidden_size) / w0
    
    initial = Siren(input_size, hidden_size, 
      w_std = 1/input_size, activation = Sine(w0_initial))
    final = Siren(hidden_size, output_size, w_std = w_std)

    layers = [Siren(hidden_size, hidden_size, w_std = w_std, activation = Sine(w0))
     for _ in range(hidden_layers)]
    self.layers = nn.Sequential(initial, *layers, final)


  def forward(self, x):
    return self.layers(x)
