from structs.torch import shape
from torch import nn
import torch
from torch.nn import functional as F
from numbers import Number


def un_batch(f, x):
  assert x.dim() == 3, f'Expected 3D input (B,C,N) (got shape {x.shape})'
  batches, channels, n = x.shape
  out = f(x.permute(0, 2, 1).reshape(batches * n, channels))

  # ((B*N), C) to (B,N,C)
  out = out.view(batches, -1, out.shape[1])
  # (B,N,C) to (B,C,N)
  return out.permute(0, 2, 1)

class UnBatch(nn.Module):
  def __init__(self, m):
    super(UnBatch, self).__init__()
    self.m = m

  def forward(self, x):
    return un_batch(self.m.forward, x)


class PerImage(nn.Module):
  def __init__(self, num_images, create_model):
    super(PerImage, self).__init__()

    self.image_models = nn.ModuleList(
        [ create_model() for _ in range(num_images) ]
    )

  def forward(self, input):
    image_index, samples = input

    def f(i, x):
      return self.image_models[i].forward(x.permute(1, 0))

    out = [f(i, x) for x, i in zip(samples, image_index)]
    return torch.stack(out, dim=0).permute(0, 2, 1)


class SplitGrid(nn.Module):
  def __init__(self, num_splits, create_model):
    super(SplitGrid, self).__init__()

    if isinstance(num_splits, Number):
      num_splits = (num_splits, num_splits)
    self.num_splits = tuple(num_splits)

    w, h = num_splits
    self.models = SplitModule([create_model() for _ in range(0, w * h)])

  def forward(self, samples):

    i, j = [ torch.floor(samples[:, d] * n).clip(0, n - 1)
      for d in [0, 1]
        for n in [self.num_splits[d]] ] 

    inds = j * self.num_splits[0] + i
    return self.models(samples, inds)
    



class SplitModule(nn.Module):
  def __init__(self, modules):
    super(SplitModule, self).__init__()
    self.split = nn.ModuleList(modules)

  def forward(self, features, inds):

    assert torch.all(inds < len(self.split)),\
      f"SplitModule.foward: {(inds > len(self.split)).sum().item()} indexes out of bounds"

    split_outputs = [m.forward(features[inds == i]) for i, m in enumerate(self.split)]
    
    outputs = torch.empty( (inds.shape[0], *split_outputs[0].shape[1:]), 
      dtype=split_outputs[0].dtype, device=split_outputs[0].device)
    for i, output in enumerate(split_outputs):
      outputs[inds == i] = output
    
    return outputs