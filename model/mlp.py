from structs.torch import shape
from torch import nn
import torch
from torch.nn import functional as F


def batchnorm_layer(inp, out):
  return nn.Sequential(
    nn.Linear(inp, out),
    nn.ReLU(),
    nn.BatchNorm1d(out)
  )


def relu_layer(inp, out):
  return nn.Sequential(
    nn.Linear(inp, out),
    nn.ReLU()
  )

class MLP(nn.Module):

  def __init__(self, input_size, hidden_size, 
    output_size=3, num_layers=4, layer_type=relu_layer):
    
    super(MLP, self).__init__()
    assert num_layers >= 3

    layers = (
      [layer_type(input_size, hidden_size)] + 
      [layer_type(hidden_size, hidden_size) for _ in range(num_layers - 2)] + 
      [nn.Linear(hidden_size, output_size)] 
    )
      
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)


class ResNet(nn.Module):

  def __init__(self, input_size, hidden_size, 
    output_size=3, num_layers=4, layer_type=relu_layer):
    
    super(ResNet, self).__init__()
    assert num_layers >= 3

    self.output = nn.Linear(hidden_size, output_size)
    self.input =  layer_type(input_size, hidden_size) 

    layers = [layer_type(hidden_size, hidden_size) 
      for _ in range(num_layers - 2)] 
      
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    x = self.input(x)
    for layer in self.layers:
      x = x + layer(x)

    return self.output(x)

