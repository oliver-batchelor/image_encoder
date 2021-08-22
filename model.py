from structs.torch import shape
from torch import nn
import torch


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
    output_size=3, num_layers=4, layer_type=batchnorm_layer):
    
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
    output_size=3, num_layers=4, layer_type=batchnorm_layer):
    
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


def with_batches(f, x):
  assert x.dim() == 2, f'Expected 3D input (B,C,N) (got shape {x.shape})'
  batches, channels, n  = x.shape
  out = f(x.permute(0, 2, 1).reshape(batches * n, channels))

  # ((B*N), C) to (B,N,C)
  out = out.view(batches, -1, out.shape[1])
  # (B,N,C) to (B,C,N)
  return out.permute(0, 2, 1)

class SplitModules(nn.Module):
  def __init__(self, ids, feature_mapper, create_model):
    super(SplitModules, self).__init__()

    self.image_models = nn.ModuleDict({
      filename:create_model() for filename in ids
    })

    self.feature_mapper = feature_mapper

  def forward(self, ids, samples):

    def f(id, x):
      return self.image_models[id].forward(self.feature_mapper(x))

    out = [ f(id, x) for x, id in zip(samples, ids)]
    return torch.stack(out, dim=0).permute(0, 2, 1)
  