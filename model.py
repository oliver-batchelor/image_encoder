from torch import nn


def batchnorm_layer(inp, out):
  return nn.Sequential(
    nn.Linear(inp, out),
    nn.ReLU(),
    nn.BatchNorm1d(out)
  )


class MLP(nn.Module):

  def __init__(self, input_size, hidden_size, output_size=3, num_layers=4, layer_type=batchnorm_layer):
    assert num_layers >= 3

    layers = (
      [layer_type(input_size, hidden_size)] + 
      [layer_type(hidden_size, hidden_size) for _ in range(num_layers - 2)] + 
      [layer_type(hidden_size, output_size)]
    )
      
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)