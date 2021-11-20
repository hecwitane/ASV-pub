import torch.nn as nn
import torch.optim as optim
from runner import Runner
from model import F34

runner = Runner()
runner.set_criterion(nn.CrossEntropyLoss)
runner.set_dataset('car', (64, 512))
runner.set_device('cuda')
runner.set_initializer('asv.backward')
runner.set_model(F34)
runner.set_optimizer(optim.Adam, lr=1e-4)
runner.model_summary()
runner.run(iterations=1000, save='result/car')
