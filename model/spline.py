import torch
from torch import nn

def bspline_coeffs(t):
  it = 1 - t
  t2 = t * t
  t3 = t2 * t

  coeffs = [
    it * it * it,
    3 * t3 - 6 * t2 + 4, 
    -3 * t3 + 3 * t2 + 3 * t + 1, 
    t3]
  return 1/6 * torch.stack(coeffs, dim=-1)  

def cubic_bspline(features, t):  
  coeffs = bspline_coeffs(t)
  return (coeffs.view(*coeffs.shape, 1) * features).sum(-2)

  
def cubic_bezier(features, t):
  it = 1 - t
  it2 = it * it
  t2 = t * t

  coeffs = torch.stack([it2 * it, 3 * it2 * t, 3 * it * t2, t2 * t], dim=-1)  
  b = coeffs.view(*coeffs.shape, 1) * features
  return b.sum(-2) 


def local_window(features, t, window = 4, wrap=False):
    num_splines, num_points, num_features = features.shape
    num_samples = t.shape[0]

    t_points = t.T * (num_points - 1 if not wrap else num_points) 
    start = torch.floor(t_points)

    ranges = torch.arange(window, device=t.device).view(1, 1, window)
    inds = (start.long().unsqueeze(2) + ranges - 1)
    
    if wrap:
      inds = inds % num_points
    else:
      inds = inds.clamp(0, num_points - 1)
    
    inds = inds.reshape(num_splines, window * num_samples, 1)
    inds = inds.expand(num_splines, window * num_samples, num_features)

    features = features.gather(1, inds)
    features = features.view(num_splines, num_samples, window, num_features)

    return features, (t_points - start)



class BSplineCubic(nn.Module):
  def __init__(self, control_points, repeating=False):
    super(BSplineCubic, self).__init__()

    self.register_buffer("features", control_points)
    self.repeating = repeating

  @staticmethod
  def randn(num_splines, num_points, scale=1.0, repeating=False):
    points = torch.randn(num_splines, num_points, 1) * scale
    return BSplineCubic(points, repeating=repeating)


  def forward(self, t):

    assert t.shape[1] == self.features.shape[0]
    features, local_t = local_window(self.features, t, window=4, wrap=self.repeating)
    
    out = cubic_bspline(features, local_t)
    return out.permute(2, 1, 0).squeeze(0)

    
if __name__ == "__main__":
  controls1 = torch.tensor([(1, 0), (0.5, 1), (1.0, -1.5), (1.5, 1), (2.0, 0)])
  controls3 = torch.tensor([(1, 0), (0.5, -1), (1.0, 1.5), (1.5, -1), (2, 0)])
  
  # spline = BSplineCubic( torch.stack([controls1, controls3]), repeating=True )
  spline = BSplineCubic.randn(2, 4, 2, scale=5, repeating=True)


  t = torch.linspace(0.0, 1.0, 100)
  ts = torch.stack([t, t], dim=1)

  import seaborn
  import matplotlib.pyplot as plt

  # b = bspline_coeffs(t)
  # for i in range(0, 4):
  #   plt.plot(t, b[:, i])

  # plt.plot(t, b.sum(1))


  coords = spline.forward(ts)
  for line, f in zip(coords, spline.features):
    x, y = line.numpy().T
    plt.plot(x, y)


    x, y = f.numpy().T
    if spline.repeating:
      x = [*x, x[0]]
      y = [*y, y[0]]
    plt.plot(x, y, marker='o')


  plt.show()
