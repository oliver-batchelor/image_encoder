from torch import nn


def batchnorm_layer(inp, out):
  return nn.Sequential(
    nn.Linear(inp, out),
    nn.ReLU(),
    nn.BatchNorm1d(out)
  )

def mlp(inp, hidden, out=3, num_layers=4, layer_type=batchnorm_layer):
  assert num_layers >= 3

  layers = (
    [layer_type(inp, hidden)] + 
    [layer_type(hidden, hidden) for _ in range(num_layers - 2)] + 
    [layer_type(hidden, out)]
  )
    
  return nn.Sequential(*layers)