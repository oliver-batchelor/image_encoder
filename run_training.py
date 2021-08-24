from model.mlp import MLP, ResNet
from model.split import PerImage, SplitGrid, UnBatch
import torch
from model.siren import SirenNet
from torch import nn
from trainer import CoordinateTrainer
from dataset import dataloaders, load_images
from functools import partial
from model.features import RandomFourier, FeatureGrids
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from structs.torch import shape


def random_fourier(num_images, mapping_size, hidden_layers=4):
  def create_model():
    return nn.Sequential(RandomFourier(2, mapping_size // 2),
                         MLP(mapping_size, mapping_size, output_size=3,
                             hidden_layers=hidden_layers),
                         nn.Sigmoid()
                         )

  # def create_model():
  #   return nn.Sequential(
  #     SirenNet(2, mapping_size, 3, num_layers),
  #     nn.Sigmoid()
  #   )

  # return PerImage(image_ids, partial(SplitGrid, (4, 4), create_model))

  return PerImage(num_images, create_model)


def features_mlp(num_images, grid_size=(16, 16),
                 feature_size=128, hidden_size=256, hidden_layers=4):

  shared = nn.Sequential(
      MLP(feature_size, hidden_size, output_size=3, hidden_layers=hidden_layers),
      nn.Sigmoid()
  )
  return nn.Sequential(
      FeatureGrids(num_images, grid_size=grid_size, num_features=feature_size),
      UnBatch(shared)
  )


def main():
  parser = ArgumentParser()
  parser.add_argument("input", default="./test_images", help="input path")

  args = parser.parse_args()

  with torch.no_grad():
    images = load_images(args.input)
    loaders = dataloaders(images, batch_size=2,
                          samples=32*1024, epoch_size=256)

    model = features_mlp(len(images), grid_size=(64, 64),
                         feature_size=128, hidden_size=256, hidden_layers=4)

    # model = random_fourier(len(images), mapping_size = 256, hidden_layers=5)

    encoder = CoordinateTrainer(model=model,
                                lr=1e-3, train_interations=40)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger("log", name="encoder")
    trainer = Trainer(logger=logger, gpus=1, precision=16,
                      callbacks=[lr_monitor])

  trainer.fit(encoder, train_dataloaders=loaders.train,
              val_dataloaders=loaders.test)


if __name__ == '__main__':
  main()
