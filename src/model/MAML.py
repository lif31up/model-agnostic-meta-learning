import torch.optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class MAML(nn.Module):
  def __init__(self, inpt_channels: int, hidn_channels: int, oupt_channels: int, config: tuple):
    super(MAML, self).__init__()
    self.conv1 = nn.Conv2d(inpt_channels, hidn_channels, kernel_size=3, padding=1, stride=1)
    self.conv2 = nn.Conv2d(hidn_channels, hidn_channels, kernel_size=3, padding=1, stride=1)
    self.pool = nn.MaxPool2d(3)
    self.l1 = nn.Linear(in_features=49284, out_features=oupt_channels)
    self.relu, self.flatten, self.softmax = nn.ReLU(), nn.Flatten(), nn.Softmax(dim=0)
    self.epochs, self.alpha = config
  # __init__

  def forward(self, x, params=None):
    if not params: params = dict(self.named_parameters())
    x = F.conv2d(x, params['conv1.weight'], bias=params['conv1.bias'], stride=1, padding=1)
    x = F.relu(x)
    x = F.conv2d(x, params['conv2.weight'], bias=params['conv2.bias'], stride=1, padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3)
    x = x.flatten()
    x = F.linear(x, weight=params['l1.weight'], bias=params['l1.bias'])
    return F.softmax(x)
  # _forward

  def inner_update(self, task, device=None):
    local_params = {name: param.clone() for name, param in self.named_parameters()}
    for _ in range(self.epochs):
      for feature, label in DataLoader(task, shuffle=True):
        pred = self.forward(feature, local_params)
        loss = nn.MSELoss()(pred, label)
        grads = torch.autograd.grad(loss, list(local_params.values()), create_graph=True)
        local_params = {name: param - (self.alpha * grad) for (name, param), grad in zip(local_params.items(), grads)}
    # for for
    return local_params
  # inner_update()
# MAMLNet