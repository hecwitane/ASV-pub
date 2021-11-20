import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import torch
import torch.nn as nn


def _calc_fan_in_out(layer):
  if isinstance(layer, nn.Linear):
    fan_in = layer.weight.shape[1]
    fan_out = layer.weight.shape[0]
    return (fan_in, fan_out)
  if isinstance(layer, nn.Conv2d):
    c_out, c_in, ker_w, ker_h = layer.weight.shape
    fan_in = c_in * ker_w * ker_h
    fan_out = c_out * ker_w * ker_h
    return (fan_in, fan_out)
  raise Exception('unknown layer type %s' % layer)


def _initialize_xavier(model):
  ret = {}
  for module in model._modules.values():
    if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
      fan_in, fan_out = _calc_fan_in_out(module)
      std = np.sqrt(2. / (fan_in + fan_out))
      nn.init.normal_(module.weight, std=std)
      ret[module] = std
    if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
      nn.init.zeros_(module.bias)
  return ret


def _initialize_kaiming(model, mode):
  ret = {}
  for module in model._modules.values():
    if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
      fan_in, fan_out = _calc_fan_in_out(module)
      if mode == 'fan_in':
        std = np.sqrt(2. / fan_in)
      else:
        assert mode == 'fan_out'
        std = np.sqrt(2. / fan_out)
      nn.init.normal_(module.weight, std=std)
      ret[module] = std
    if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
      nn.init.zeros_(module.bias)
  return ret


_WEIGHT_LAYERS = (nn.Conv2d, nn.Linear)
_ACTIVATION_LAYERS = (nn.ReLU, )
_MAXPOOL_LAYERS = (nn.MaxPool2d, nn.AdaptiveMaxPool2d)
_AVGPOOL_LAYERS = (nn.AvgPool2d, nn.AdaptiveAvgPool2d)


def _count_conv_connections(conv, in_shape, out_shape):
  assert isinstance(conv, nn.Conv2d)
  kappa_w, kappa_h = conv.kernel_size
  p_w, p_h = conv.padding
  s_w, s_h = conv.stride
  _, w, h = in_shape
  d = conv.in_channels
  d_u = conv.out_channels
  w_u = (w + 2 * p_w - kappa_w) // s_w + 1
  h_u = (h + 2 * p_h - kappa_h) // s_h + 1
  assert (d, w, h) == in_shape
  assert (d_u, w_u, h_u) == out_shape
  ret = 0
  for i in range(w_u):
    for j in range(h_u):
      lam_w_i = p_w - s_w * i
      lam_h_j = p_h - s_h * j
      ret_l = max(0, min(kappa_w, w + lam_w_i) - max(0, lam_w_i))
      ret_m = max(0, min(kappa_h, h + lam_h_j) - max(0, lam_h_j))
      ret += ret_l * ret_m * d
  return ret * d_u


def _initialize_asv(model, input_shape, mode, std_factor=3.0):
  device = next(model.parameters()).device
  data = torch.zeros((2, *input_shape)).to(device)
  in_shapes = {}
  out_shapes = {}
  groups = []
  stack = []
  for module in model._modules.values():
    # get shape
    in_shapes[module] = data.shape[1:]
    data = module.forward(data)
    out_shapes[module] = data.shape[1:]
    # parse arch
    if len(stack) == 0:
      if isinstance(module, _WEIGHT_LAYERS):
        stack.append(module)
      continue
    if len(stack) == 1:
      if isinstance(module, (*_WEIGHT_LAYERS, *_MAXPOOL_LAYERS, *_AVGPOOL_LAYERS)):
        groups.append(stack)
        stack = [module]
      elif isinstance(module, nn.ReLU):
        stack.append(module)
        if isinstance(module, nn.Linear):
          groups.append(stack)
          stack = []
      continue
    if len(stack) == 2:
      if isinstance(module, _WEIGHT_LAYERS):
        groups.append(stack)
        stack = [module]
        continue
      if isinstance(module, (*_MAXPOOL_LAYERS, *_AVGPOOL_LAYERS)):
        stack.append(module)
      groups.append(stack)
      stack = []
      continue
    raise Exception('invalid state')
  if len(stack) > 0:
    groups.append(stack)
  # calculate initialization parameters
  target_params = []
  for group in groups:
    param = {
        'M_in': np.prod(in_shapes[group[0]]),
        'M_out_conv': np.prod(out_shapes[group[0]]),
        'eps': None,
        'tau': None,
        'gamma': None
    }
    # linear
    if isinstance(group[0], nn.Linear):
      param['eps'] = group[0].in_features * group[0].out_features
      if len(group) == 1:
        param['tau'] = param['gamma'] = 1.
      elif len(group) == 2:
        param['tau'] = param['gamma'] = .5
    # conv
    elif isinstance(group[0], nn.Conv2d):
      param['eps'] = _count_conv_connections(group[0], in_shapes[group[0]], out_shapes[group[0]])
      if len(group) == 1:
        param['tau'] = param['gamma'] = 1.
      elif len(group) == 2:
        param['tau'] = param['gamma'] = .5
      elif len(group) == 3:
        # get pooling size
        pool = group[2]
        if hasattr(pool, 'kernel_size'):
          kernel_size = pool.kernel_size
        else:
          # support global pooling only
          assert pool.output_size == (1, 1)
          kernel_size = in_shapes[pool][1:]
        # get T_ell
        if isinstance(kernel_size, tuple):
          T_ell = np.prod(kernel_size)
        else:
          T_ell = kernel_size ** 2
        if isinstance(pool, _MAXPOOL_LAYERS):
          param['tau'] = T_ell * quad(lambda s: s**2 * norm.pdf(s)
                                      * norm.cdf(s)**(T_ell - 1), 0, np.inf)[0]
          power = 2**T_ell
          param['gamma'] = (power - 1) / (T_ell * power)
        elif isinstance(pool, _AVGPOOL_LAYERS):
          param['tau'] = .5 / T_ell * (1. + (T_ell - 1.) / np.pi)
          param['gamma'] = .5 / (T_ell**2)
    if param['tau'] is None and param['gamma'] is None:
      raise Exception('unexpected architecture')
    target_params.append(param)
  # init
  target_std = {}
  for idx, group in enumerate(groups):
    param = target_params[idx]
    if mode == 'forward':
      if idx == 0:
        tau_prev = 1.
      else:
        tau_prev = target_params[idx - 1]['tau']
      M_out_conv = param['M_out_conv']
      eps = param['eps']
      std = np.sqrt(M_out_conv / (tau_prev * eps))
      base_std = np.sqrt(M_out_conv / (.5 * eps))
    elif mode == 'backward':
      M_in = param['M_in']
      eps = param['eps']
      gamma = param['gamma']
      std = np.sqrt(M_in / (gamma * eps))
      base_std = np.sqrt(M_in / (.5 * eps))
    module = group[0]
    if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
      if std_factor is not None:
        std = min(std, float(std_factor) * base_std)
      nn.init.normal_(module.weight, std=std)
      target_std[module] = std
    if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
      nn.init.zeros_(module.bias)
  return target_std


def initialize(model, method, input_shape, *args, **kwargs):
  assert isinstance(model, nn.Sequential)
  if method == 'xavier':
    return _initialize_xavier(model, *args, **kwargs)
  if method == 'kaiming.forward':
    return _initialize_kaiming(model, 'fan_in', *args, **kwargs)
  if method == 'kaiming.backward':
    return _initialize_kaiming(model, 'fan_out', *args, **kwargs)
  if method == 'asv.forward':
    return _initialize_asv(model, input_shape, 'forward', *args, **kwargs)
  if method == 'asv.backward':
    return _initialize_asv(model, input_shape, 'backward', *args, **kwargs)
  raise Exception('unknown initialization method: %s' % method)
