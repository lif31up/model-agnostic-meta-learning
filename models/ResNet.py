import torch.optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import Config


class ResNet_MAML(nn.Module):
  def __init__(self, config: Config):
    super(ResNet_MAML, self).__init__()
    self.config = config
    self.convs = self._create_convs(self.config.n_convs)
    self.act = nn.SiLU()
    self.flat = nn.Flatten(start_dim=1)
    self.pool = nn.MaxPool2d(stride=1, kernel_size=3)
    self.fc = self._get_fc(self.config.dummy)
  # __init__

  def _create_convs(self, n_convs):
    layers = nn.ModuleList()
    layers.append(
      nn.Conv2d(
        in_channels=self.config.input_channels,
        out_channels=self.config.hidden_channels,
        kernel_size=self.config.kernel_size,
        stride=self.config.stride,
        padding=self.config.padding)
    ) # first conv
    for i in range(n_convs - 1):
      layers.append(
        nn.Conv2d(
          in_channels=self.config.hidden_channels,
          out_channels=self.config.hidden_channels,
          kernel_size=self.config.kernel_size,
          padding=self.config.padding,
          stride=self.config.stride,
          bias=self.config.bias),
      ) # hidden convs
    return layers
  # _create_convs

  def forward(self, x, params=None):
    if params is None: params = dict(self.named_parameters())
    x = F.conv2d(
      input=x,
      weight=params[f'convs.{0}.weight'],
      bias=params[f'convs.{0}.bias'],
      stride=self.config.stride,
      padding=self.config.padding
    )  # first conv
    x = self.act(x)
    for i in range(1, self.convs.__len__()):
      res = x
      x = F.conv2d(
        input=x,
        weight=params[f'convs.{i}.weight'],
        bias=params[f'convs.{i}.bias'],
        stride=self.config.stride,
        padding=self.config.padding,
      )  # hidden convs
      x = self.act(x)
      x += res
    x = self.pool(x)
    x = self.flat(x)
    return F.linear(x, weight=params['fc.weight'], bias=params['fc.bias'])
  # forward

  def inner_update(self, task, device=None):
    local_param = {name: param.clone() for name, param in self.named_parameters()}  # init local params
    for _ in range(self.config.iterations): # update local params to the task
      for feature, label in DataLoader(task, batch_size=self.config.batch_size, shuffle=True, pin_memory=True, num_workers=4):
        feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
        pred = self.forward(feature, local_param)
        loss = nn.MSELoss()(pred, label)
        grads = torch.autograd.grad(loss, list(local_param.values()), create_graph=True)
        local_param = {name: param - (self.config.alpha * grad) for (name, param), grad in zip(local_param.items(), grads)}
    return local_param
  # inner_update

  def _get_fc(self, dummy):
    with torch.no_grad():
      for conv in self.convs: dummy = conv(dummy)
      dummy = self.pool(dummy)
      dummy = self.flat(dummy)
      return nn.Linear(in_features=dummy.shape[1], out_features=self.config.output_channels, bias=self.config.bias)
  # _get_fcc
# MAMLNet

class ResNet_BOIL(ResNet_MAML):
  def inner_update(self, task, device=None):
    local_params = {name: param.clone() for name, param in self.named_parameters()}  # init local params
    for _ in range(self.config.iterations):  # update local params to the task
      for feature, label in DataLoader(
          dataset=task,
          batch_size=self.config.batch_size,
          shuffle=True,
          pin_memory=True,
          num_workers=4):
        feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
        pred = self.forward(feature, local_params)
        loss = nn.MSELoss()(pred, label)
        grads = torch.autograd.grad(loss, list(local_params.values()), create_graph=True)
        local_params = {
          name: param - self.config.alpha * grad if not name.startswith('fc') else param for (name, param), grad in zip(local_params.items(), grads)}  # freezing the last layer
    return local_params
  # inner_update
# ResNetBOIL

class ResNet_ANIL(ResNet_MAML):
  def inner_update(self, task, device=None):
    local_params = {name: param.clone() for name, param in self.named_parameters()}  # init local params
    for _ in range(self.config.iterations):  # update local params to the task
      for feature, label in DataLoader(
          dataset=task,
          batch_size=self.config.batch_size,
          shuffle=True,
          pin_memory=True,
          num_workers=4):
        feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
        pred = self.forward(feature, local_params)
        loss = nn.MSELoss()(pred, label)
        grads = torch.autograd.grad(loss, list(local_params.values()), create_graph=True)
        local_params = {
          name: param - self.config.alpha * grad if name.startswith('fc') else param for (name, param), grad in
          zip(local_params.items(), grads)}  # freezing the convs
    return local_params
  # inner_update
# ResNet_ANIL