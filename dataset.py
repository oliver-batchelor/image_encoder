from os import path
import os
from natsort.natsort import natsorted
from structs.struct import struct, to_dicts, to_structs
from structs.torch import shape
import torch
import cv2

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate



image_extensions = ['jpg', 'jpeg', 'png', 'ppm', 'bmp']

def has_extension(extensions, filename):
    return any(filename.lower().endswith("." + extension) for extension in extensions)

def find_images(filepath, extensions=image_extensions):
    return [filename for filename in natsorted(os.listdir(filepath))
                if has_extension(extensions, path.join(filepath, filename))]


def create_grid(w, h):
    y, x = torch.meshgrid([
      torch.linspace(0, 1, steps=h), 
      torch.linspace(0, 1, steps=w)
    ])

    return torch.stack([y, x], dim=0)

def escape(filename):
  return filename.replace('.', '_')


def load_image(base_path, filename, device="cpu"):
  image = cv2.imread(path.join(base_path, filename), cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  grid = create_grid(image.shape[1], image.shape[0])

  return struct (
    image = torch.from_numpy(image).permute(2, 0, 1).to(device),
    grid = grid.to(device),
    filename = filename,
    id = escape(filename),
    
  )

def load_images(image_path):
  assert path.isdir(image_path)
  image_filenames = find_images(image_path)
  return [load_image(image_path, image_file)
    for image_file in image_filenames]

class Images(Dataset):
  def __init__(self, images):
    super(Images, self).__init__()
    self.images = images

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    item = self.images[index]

    _, h, w = item.image.shape
    image = item.image.view(3, -1)

    return struct(
      colors = image.float() / 255,
      grid = item.grid.view(2, -1),
      id = item.id,
      image_size = torch.LongTensor([w, h])
    )


class SampledImages(Dataset):
  def __init__(self, images, samples=128*1024):
    super(SampledImages, self).__init__()
    self.images = images
    self.samples = samples

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    item = self.images[index]
    
    image = item.image.view(3, -1)
    grid = item.grid.view(2, -1)

    samples = torch.randint(0, image.shape[1], (self.samples,))

    return struct(
      colors = image[:, samples].float() / 255,
      grid = grid[:, samples],
      id = item.id
    )


class Repeat(Dataset):
    def __init__(self, dataset, num_repeats=1024):
      super(Repeat, self).__init__()
      self.dataset = dataset
      self.num_repeats = num_repeats
  
    def __len__(self):
      return len(self.dataset) * self.num_repeats

    def __getitem__(self, index):
      return self.dataset[index % len(self.dataset)]



def dataloaders(images, batch_size=4, samples=32*1024, epoch_size=64):
  train_dataset = Repeat(SampledImages(images, samples=samples), epoch_size)

  training = DataLoader(train_dataset, batch_size=batch_size, 
    num_workers=4, pin_memory=True)

  testing = DataLoader(Images(images), batch_size=1, num_workers=4)
  return struct(train=training, test=testing)