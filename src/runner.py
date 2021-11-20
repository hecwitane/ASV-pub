import os
import gc
import random
import time
from datetime import datetime
import pickle
import platform
import numpy as np
import torch
import torch.nn as nn
from dataset import load_dataset
from initializer import initialize


class Runner:
  def __init__(self):
    self.run_time = None
    self.current = None
    self.losses = []
    self.scores = []

  def set_model(self, model, *args, **kwargs):
    self.model_class = model
    self.model_args = args
    self.model_kwargs = kwargs

  def set_initializer(self, method, *args, **kwargs):
    self.init_method = method
    self.init_method_args = args
    self.init_method_kwargs = kwargs

  def set_optimizer(self, optim, *args, **kwargs):
    self.optimizer_class = optim
    self.optimizer_args = args
    self.optimizer_kwargs = kwargs

  def set_criterion(self, crit, *args, **kwargs):
    self.criterion_class = crit
    self.criterion_args = args
    self.criterion_kwargs = kwargs

  def set_dataset(self, name, batch_size, num_workers=None):
    self.dataset_name = name
    self.dataset_batch_size = batch_size
    if num_workers is None:
      num_workers = os.cpu_count() - 1
    self.dataset_workers = num_workers

  def set_device(self, device):
    self.device = device

  def init_seed(self, seed):
    # cf. https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668,
    #     https://pytorch.org/docs/stable/notes/randomness.html
    print('set seed =', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  def init(self, seed):
    # clear mems
    torch.cuda.empty_cache()
    gc.collect()

    # reset RNGs seed
    self.init_seed(seed)

    # load dataset and estimate input shape
    tl, vl = load_dataset(self.dataset_name,
                          batch_size=self.dataset_batch_size,
                          num_workers=self.dataset_workers,
                          seed=seed)

    # estiname input shape and n_cls
    ds = tl.dataset
    input_shape = input_shape = ds[0][0].shape
    n_cls = len(ds.classes)

    # select device
    if isinstance(self.device, int):
      # select GPU device
      if torch.cuda.is_available():
        torch.cuda.set_device(self.device)
        device = torch.device('cuda')
      else:
        device = torch.device('cpu')
    else:
      device = torch.device(self.device)
    if device.type == 'cuda':
      print('use GPU:', torch.cuda.get_device_name(device.index))
    else:
      print('use CPU:', platform.processor())

    # initialize model
    if 'out_features' not in self.model_kwargs:
      self.model_kwargs['out_features'] = n_cls
    model = self.model_class(*self.model_args, **self.model_kwargs).to(device)
    init_std = initialize(model, self.init_method, input_shape, *self.init_method_args, **self.init_method_kwargs)

    # initialize optimizer
    optim = self.optimizer_class(model.parameters(), *self.optimizer_args, **self.optimizer_kwargs)
    crit = self.criterion_class(*self.criterion_args, **self.criterion_kwargs)

    self.current = {
        'seed': seed,
        'device': device,
        'model': model,
        'input_shape': input_shape,
        'n_cls': n_cls,
        'init_std': init_std,
        'optimizer': optim,
        'criterion': crit,
        'train_loader': tl,
        'val_loader': vl,
    }

  def run_train(self):
    # load variables
    c = self.current
    model = c['model']
    device = c['device']
    optimizer = c['optimizer']
    criterion = c['criterion']
    train_loader = c['train_loader']
    # train
    model.train()
    losses = []
    batches = len(train_loader)
    for i, (data, label) in enumerate(train_loader):
      data = data.to(device)
      label = label.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, label)
      loss_value = loss.detach().cpu().item()
      print('\r[%d/%d] train loss: %f\033[K' %
            (i + 1, batches, loss_value), end='')
      losses.append(loss_value)
      loss.backward()
      optimizer.step()
      del data, label, output, loss
    print('')
    return losses

  def run_val(self):
    # load variables
    c = self.current
    model = c['model']
    device = c['device']
    criterion = c['criterion']
    val_loader = c['val_loader']
    # val
    model.eval()
    batches = len(val_loader)
    total_loss = 0
    correct = 0
    with torch.no_grad():
      for i, (data, label) in enumerate(val_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        total_loss += loss.detach().cpu().item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).detach().sum().cpu().item()
        print('\r[%d/%d] calculating val score...' % (i + 1, batches), end='')
        del data, label, output, loss, pred
    average_loss = total_loss / batches
    acc = correct / len(val_loader.dataset) * 100
    print('\rval loss: %f, val acc: %.4f %%\033[K' % (average_loss, acc))
    return (average_loss, acc)

  def run_iterations(self, iterations):
    loss_history = []
    score_history = []
    start = datetime.now()
    for iteration in range(iterations):
      print('iteration %d/%d' % (iteration + 1, iterations))
      loss_history.append(self.run_train())
      score_history.append(self.run_val())
      print('peak val acc: %.4f %%' % np.array(score_history)[:, 1].max())
      print('elapsed %s sec per iteration' %
            str((datetime.now() - start) / (iteration + 1)).split('.')[0])
    return (loss_history, score_history)

  def run(self, iterations, samples=1, seed=0, save=False):
    self.run_time = int(time.time())
    self.losses = []
    self.scores = []
    for i in range(samples):
      print('samples %d/%d' % (i + 1, samples))
      self.init(seed + i)
      loss, score = self.run_iterations(iterations)
      self.losses.append(loss)
      self.scores.append(score)
      if save is not False:
        if isinstance(save, str):
          self.save(save)
        else:
          self.save()

  def model_summary(self):
    # init
    self.init(0)
    c = self.current
    model = c['model']
    train_loader = c['train_loader']
    val_loader = c['val_loader']
    input_shape = c['input_shape']
    n_cls = c['n_cls']
    criterion = c['criterion']
    optimizer = c['optimizer']
    init_std = c['init_std']
    # print
    print('-' * 80)
    print('Dataset:', self.dataset_name)
    print('  num_workers:', self.dataset_workers)
    print('  input shape:', tuple(input_shape))
    print('  classes:', n_cls)
    print('  train samples:', len(train_loader.dataset))
    print('  train batch size', train_loader.batch_size)
    print('  val samples:', len(val_loader.dataset))
    print('  val batch size', val_loader.batch_size)
    print('Model: %s' % model.__class__.__name__)
    device = next(model.parameters()).device
    data = torch.zeros((2, *input_shape)).to(device)
    print(tuple(data.shape[1:]), 'Input')
    for module in model._modules.values():
      data = module.forward(data)
      print(tuple(data.shape[1:]), module)
    count_params = np.sum([p.numel()
                           for p in model.parameters() if p.requires_grad])
    print('trainable parameters: %e' % count_params)
    print('weight layers:', len(init_std.keys()))
    print('criterion:', criterion)
    print('optimizer:', optimizer)
    print('init:', self.init_method)
    for module in init_std:
      print('  %.6f %s...' % (init_std[module], str(module)[:60]))
    print('-' * 80)

  def save(self, save_dir='result'):
    if self.run_time is None:
      return
    save_path = '%s/%s/%s/%s/%s_%d.pickle' % (
        save_dir,
        self.dataset_name,
        self.model_class.__name__,
        self.optimizer_class.__name__,
        self.init_method,
        self.run_time
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as fp:
      current = self.current
      self.current = {}
      for key in current:
        self.current[key] = str(current[key])
      pickle.dump(self, fp)
      self.current = current
