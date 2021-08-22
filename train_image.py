from trainer import CoordinateEncoder
from dataset import dataloaders, load_images
from functools import partial
from features import RandomFourier
from argparse import ArgumentParser
from model import MLP, SplitModules

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def random_fourier(image_ids, mapping_size, num_layers=4):
  feature_mapper = RandomFourier(2, mapping_size // 2)
  create_mlp = partial(MLP, mapping_size, mapping_size, output_size=3, num_layers=num_layers)
  return SplitModules(image_ids, feature_mapper, create_mlp)




def main():
  parser = ArgumentParser()
  parser.add_argument("input", default="./test_images", help="input path")

  args = parser.parse_args()
  images = load_images(args.input)
  
  loaders = dataloaders(images, batch_size=4, samples=32*1024, epoch_size=128)
  encoder = CoordinateEncoder(model = 
    random_fourier([image.id for image in images], mapping_size = 256, num_layers=5))


  logger = TensorBoardLogger("log", name="encoder")
  trainer = Trainer(logger=logger, gpus=1, precision=16)

  trainer.fit(encoder, train_dataloaders=loaders.train, val_dataloaders=loaders.test)

if __name__ == '__main__':
  main()
