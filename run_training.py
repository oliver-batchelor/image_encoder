from model.mlp import MLP, ResNet
from model.split import PerImage, SplitGrid, UnBatch
from model import Sine

import torch
from model.siren import SirenNet
from torch import nn
from trainer import CoordinateTrainer
from dataset import dataloaders, load_images
from functools import partial
from model.features import Concatenated, FeatureGrids, Modulated, positional_fourier, linear_fourier, random_fourier
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from structs.torch import shape


def mlp_model(num_images, mapping_size, hidden_layers=4):
  def create_model():
    input_size = 32
    # encoder = random_fourier(2, mapping_size // 2)
    encoder = linear_fourier(2, input_size, num_phases=2)

    return nn.Sequential(encoder,
                         MLP(encoder.num_outputs, mapping_size, output_size=3,
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


def features_mlp(num_images, grid_size=(1, 1),
                 feature_size=128, hidden_size=256, hidden_layers=4):


  # fourier_encoding = positional_fourier(2, 8, num_phases=2)
  fourier_encoding = linear_fourier(2, 16)


  print(f"N features {fourier_encoding.num_outputs}")
  feature_encoder = Concatenated(
    FeatureGrids(num_images, grid_size=grid_size, 
      num_features=feature_size, interpolation='bicubic'),
    fourier_encoding
  )

  shared = nn.Sequential(
      MLP(feature_encoder.num_outputs, hidden_size, output_size=3, hidden_layers=hidden_layers),
      nn.Sigmoid()
  )

  return nn.Sequential(
      feature_encoder,
      UnBatch(shared)
  )


def main():
  parser = ArgumentParser()
  parser.add_argument("input", default="./test_images", help="input path")

  args = parser.parse_args()

  with torch.no_grad():
    images = load_images(args.input)
    loaders = dataloaders(images, batch_size=4,
                          samples=16*1024, epoch_size=256)

    model = features_mlp(len(images), grid_size=(16, 16),
                         feature_size=128, hidden_size=512, hidden_layers=5)

    # model = mlp_model(len(images), mapping_size = 256, hidden_layers=5)

    encoder = CoordinateTrainer(model=model,
                                lr=1e-3, train_interations=100)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger("log", name="encoder")
    trainer = Trainer(logger=logger, gpus=1, precision=16,
                      callbacks=[lr_monitor])

  trainer.fit(encoder, train_dataloaders=loaders.train,
              val_dataloaders=loaders.test)


if __name__ == '__main__':
  main()
