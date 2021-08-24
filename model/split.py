from structs.torch import shape
from torch import nn
import torch
from torch.nn import functional as F
from numbers import Number


def with_batches(f, x):
  assert x.dim() == 2, f'Expected 3D input (B,C,N) (got shape {x.shape})'
  batches, channels, n = x.shape
  out = f(x.permute(0, 2, 1).reshape(batches * n, channels))

  # ((B*N), C) to (B,N,C)
  out = out.view(batches, -1, out.shape[1])
  # (B,N,C) to (B,C,N)
  return out.permute(0, 2, 1)


class PerImage(nn.Module):
  def __init__(self, ids, create_model):
    super(PerImage, self).__init__()

    self.image_models = nn.ModuleDict({
        filename: create_model() for filename in ids
    })

  def forward(self, ids, samples):

    def f(id, x):
      return self.image_models[id].forward(x.permute(1, 0))

    out = [f(id, x) for x, id in zip(samples, ids)]
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
    


def group_by_indexes(f, inds):
  _, sort_inds, counts = torch.unique(inds, return_counts=True, return_inverse=True)
  fs = torch.split_with_sizes(f[sort_inds], counts.tolist())
  return fs, sort_inds

def ungroup_indexes(fs, sort_inds):
  sorted_out =  torch.cat(fs)
  outputs = torch.empty(sorted_out.shape, dtype=sorted_out.dtype, device=sorted_out.device)
  outputs[sort_inds] = sorted_out

  return outputs

class SplitModule(nn.Module):
  def __init__(self, modules):
    super(SplitModule, self).__init__()
    self.split = nn.ModuleList(modules)

  def forward(self, features, inds):
    grouped_features, sort_inds = group_by_indexes(features, inds)

    assert len(grouped_features) == len(self.split),\
      f"SplitModule.foward: {(inds > len(self.split)).sum().item()} indexes out of bounds"

    outputs =  [m.forward(f) for f, m in zip(grouped_features, self.split)]
    return ungroup_indexes(outputs, sort_inds)