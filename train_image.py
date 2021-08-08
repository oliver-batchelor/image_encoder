from torch.utils.data.dataset import Dataset
from features import GaussianFourierFeatures
import torch
import os

from torch import nn
from torch.utils.data import DataLoader

import tqdm
import cv2

from argparse import ArgumentParser
from structs import struct

from os import path
from natsort import natsorted

from model import MLP, mlp


image_extensions = ['jpg', 'jpeg', 'png', 'ppm', 'bmp']

def has_extension(extensions, filename):
    return any(filename.lower().endswith("." + extension) for extension in extensions)

def find_images(filepath, extensions=image_extensions):
    return [path.join(filepath, filename) for filename in natsorted(os.listdir(filepath))
                if has_extension(extensions, path.join(filepath, filename))]


def psnr(mse): 
  return -10 * torch.log10(mse)


def train_model(model, learning_rate,  training, iters=100):

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_psnrs = []

    for _ in range(iters):
      for data in training:

          model.train()
          optim.zero_grad()

          output = model(data.grid)
          loss = loss_fn(output, data.labels)

          loss.backward()
          optim.step()

          train_psnrs.append(psnr(loss).item())


def create_grid(w, h):
    y, x = torch.meshgrid([
      torch.linspace(0, 1, steps=h), 
      torch.linspace(0, 1, steps=w)
    ])

    return torch.stack([y, x], dim=-1)


def load_image(filename, device="cpu"):
  image = cv2.imread(filename, cv2.IMREAD_COLOR)
  grid = create_grid(image.shape[1], image.shape[0])

  return dict (
    image = image.to(device),
    grid = grid.to(device),
    filename = filename
  )

class ImageDataset(Dataset):
  def __init__(self, images):
    super(ImageDataset, self).__init__()
    self.images = images

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
      return self.images[index]


def main():
  parser = ArgumentParser()
  parser.add_argument("input", default="./test_images", help="input path")

  args = parser.parse_args()
  device = "cuda:0"

  assert path.isdir(args.input)
  images = [load_image(image) for image in find_images(args.input)]

  mapping_size = 256

  model = nn.Sequential(
    GaussianFourierFeatures(2, mapping_size),
    MLP(mapping_size, mapping_size, output_size=3, num_layers=4)
  )

  training = DataLoader(ImageDataset(images), batch_size=1, num_workers=1)
  train_model(model, training, learning_rate=1e-4, iters=250, device=device)  


if __name__ == '__main__':
  main()
