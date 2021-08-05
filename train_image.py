import torch
import os

from torch import nn
import tqdm
import cv2

from argparse import ArgumentParser
from structs import struct

from os import path
from natsort import natsorted

image_extensions = ['jpg', 'jpeg', 'png', 'ppm', 'bmp']

def has_extension(extensions, filename):
    return any(filename.lower().endswith("." + extension) for extension in extensions)

def find_images(filepath, extensions=image_extensions):
    return [path.join(filepath, filename) for filename in natsorted(os.listdir(filepath))
                if has_extension(extensions, path.join(filepath, filename))]



def psnr(mse): 
  return -10 * torch.log10(mse)


def train_model(model, learning_rate,  images, iters=10, device='cuda:0'):

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_psnrs = []
    xs = []

    for data in images:
        model.train()
        optim.zero_grad()

        output = model(data.grid)
        loss = loss_fn(output, data.labels)

        loss.backward()
        optim.step()

        train_psnrs.append(psnr(loss))

    # cv2.imwrite(f"imgs/{i}.jpeg", v_o.permute(0, 3, 1, 2))


def create_grid(w, h):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h), torch.linspace(0, 1, steps=w)])
    return torch.stack([grid_y, grid_x], dim=-1)


def load_image(filename, device="cpu"):
  image = cv2.imread(filename, cv2.IMREAD_COLOR)
  grid = create_grid(image.shape[1], image.shape[0])

  return struct (
    image = image.to(device),
    grid = grid.to(device),
    filename = filename
  )

def main():
  parser = ArgumentParser()
  parser.add_argument("input", help="input path")

  args = parser.parse_args()

  device = "cuda:0"

  learning_rate = 1e-4
  iters = 250

  mapping_size = 256

  assert path.isdir(args.input)
  images = find_images(args.input)

  output = train_model(learning_rate, iters, device=device)  


if __name__ == '__main__':
  main()
