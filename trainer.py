
from structs.struct import Struct, struct
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from structs.torch import shape

import pytorch_lightning as pl
from operator import add
from functools import reduce
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR


def psnr(mse): 
  return -10 * torch.log10(mse)

class CoordinateTrainer(pl.LightningModule):
  def __init__(self, model, lr=1e-3, train_interations=20):
    super(CoordinateTrainer, self).__init__()
    self.model = model
    self.lr = lr
    self.train_iterations = train_interations

  def forward(self, index, grids):
    return self.model.forward( (index, grids) )

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.1**(1/self.train_iterations), last_epoch=-1, verbose=False)
    scheduler = {
        'scheduler': scheduler,
        'interval': 'epoch',
        'frequency': 1,
    }

    return [optimizer], [scheduler]

  def compare(self, batch):
    output = self.forward(batch.index, batch.grid)
    loss = F.mse_loss(output, batch.colors)
    return output, loss

  def training_step(self, train_batch, batch_idx):
    train_batch = Struct(train_batch)
    _, loss = self.compare(train_batch)
    self.log('train_psnr', psnr(loss).item(), prog_bar=True)
    return loss

  def validation_step(self, val_batch, batch_idx):
    val_batch = Struct(val_batch)
    assert val_batch.grid.shape[0] == 1
    
    output, loss = self.compare(val_batch)
    id = val_batch.filename[0]

    result = struct(
      psnr = psnr(loss).item(),
      mse = loss.item()
    )

    self.log(f'val_psnr_{id}', result.psnr)

    w, h = val_batch.image_size[0]
    image = output.view(3, h.item(), w.item())

    self.log_image(id, image)
    return result

  def validation_epoch_end(self, results):
    total = reduce(add, results) / len(results)

    self.log(f'val_psnr', total.psnr, prog_bar=True)
    self.log(f'val_mse', total.mse)


  def log_image(self, name, image):
    image = image.clip(min=0, max=1)

    tensorboard = self.logger.experiment
    tensorboard.add_image(name, image, self.current_epoch)
    
