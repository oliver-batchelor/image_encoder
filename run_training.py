from model.mlp import MLP
from model.split import PerImage, SplitGrid
import torch
from model.siren import SirenNet
from torch import nn
from trainer import CoordinateTrainer
from dataset import dataloaders, load_images
from functools import partial
from model.features import RandomFourier
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def random_fourier(image_ids, mapping_size, num_layers=4):
  def create_model():
    return nn.Sequential(RandomFourier(2, mapping_size // 2),
      MLP(mapping_size, mapping_size, output_size=3, num_layers=num_layers),
      nn.Sigmoid()
    )

  # def create_model():
  #   return nn.Sequential(
  #     SirenNet(2, mapping_size, 3, num_layers),
  #     nn.Sigmoid()
  #   )


  # return PerImage(image_ids, partial(SplitGrid, (4, 4), create_model))

  return PerImage(image_ids, create_model)



def main():
  parser = ArgumentParser()
  parser.add_argument("input", default="./test_images", help="input path")

  args = parser.parse_args()

  with torch.no_grad():
    images = load_images(args.input)
    
    loaders = dataloaders(images, batch_size=4, samples=32*1024, epoch_size=256)
    encoder = CoordinateTrainer(model = 
      random_fourier([image.id for image in images], mapping_size = 256, num_layers=4),
      lr = 1e-3)


    logger = TensorBoardLogger("log", name="encoder")
    trainer = Trainer(logger=logger, gpus=1, precision=16)

  trainer.fit(encoder, train_dataloaders=loaders.train, val_dataloaders=loaders.test)

if __name__ == '__main__':
  main()
