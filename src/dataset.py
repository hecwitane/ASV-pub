from os import path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_dataset(name, batch_size, root=None, num_workers=0, seed=0):
  if root is None:
    root = path.join(path.dirname(__file__), '..', 'dataset')

  def __worker_set_seed(worker_id):
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)

  datasetNames = [
      # subset of VMMRdb
      # http://vmmrdb.cecsresearch.org/
      'car',
      # subset of FoodX-251
      # https://github.com/karansikka1/iFood_2019/
      'food',
      # subset of iNaturalist 2019
      # https://github.com/visipedia/inat_comp
      'fungi'
  ]
  if name not in datasetNames:
    raise Exception('unknown dataset: %s' % name)

  dataset_train = datasets.ImageFolder(
      path.join(root, name, 'train'), transform=transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ]))
  dataset_val = datasets.ImageFolder(
      path.join(root, name, 'val'), transform=transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ]))

  train_bs, val_bs = batch_size
  return (DataLoader(dataset_train, train_bs, shuffle=True, num_workers=num_workers, worker_init_fn=__worker_set_seed),
          DataLoader(dataset_val, val_bs, shuffle=False, num_workers=num_workers, worker_init_fn=__worker_set_seed))
