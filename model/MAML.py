import torch.optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class MAML(nn.Module):
  def __init__(self, config):
    super(MAML, self).__init__()
    self.config = config
    self.conv1 = nn.Conv2d(self.config["input_channels"], self.config["hidden_channels"], kernel_size=self.config["conv:kernel_size"], padding=self.config["conv:padding"], stride=self.config["conv:stride"])
    self.conv2 = nn.Conv2d(self.config["hidden_channels"], self.config["hidden_channels"], kernel_size=self.config["conv:kernel_size"], padding=self.config["conv:padding"], stride=self.config["conv:stride"])
    self.conv3 = nn.Conv2d(self.config["hidden_channels"], self.config["hidden_channels"], kernel_size=self.config["conv:kernel_size"], padding=self.config["conv:padding"], stride=self.config["conv:stride"])
    self.pool = nn.MaxPool2d(kernel_size=3)
    self.act, self.flatten = nn.ReLU(), nn.Flatten(start_dim=1)
    self.l1 = nn.Linear(in_features=self.config["l1_in_features"], out_features=self.config["output_channels"])
  # __init__

  def forward(self, x, params=None):
    if not params: params = dict(self.named_parameters())  # uses meta/global params when local params not given
    x = F.conv2d(x, weight=params['conv1.weight'], bias=params['conv1.bias'], padding=self.config["conv:padding"], stride=self.config["conv:stride"])
    res = x
    x = F.conv2d(self.act(x) + res, weight=params['conv2.weight'], bias=params['conv2.bias'], padding=self.config["conv:padding"], stride=self.config["conv:stride"])
    res = x
    x = F.conv2d(self.act(x) + res, weight=params['conv3.weight'], bias=params['conv3.bias'], padding=self.config["conv:padding"], stride=self.config["conv:stride"])
    x = self.pool(x)
    x = self.flatten(x)
    return F.linear(x, weight=params['l1.weight'], bias=params['l1.bias'])
  # _forward

  def inner_update(self, task, config, device=None):
    """it retuns adapted/inner-updated params which associated with given task"""
    local_params = {name: param.clone() for name, param in self.named_parameters()}  # init local params
    for _ in range(config["iterations"]): # update local params to the task
      for feature, label in DataLoader(task, batch_size=config["iterations:batch_size"], shuffle=True, pin_memory=True, num_workers=4):
        feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
        pred = self.forward(feature, local_params)
        loss = nn.MSELoss()(pred, label)
        grads = torch.autograd.grad(loss, list(local_params.values()), create_graph=True)
        local_params = {name: param - (config["alpha"] * grad) for (name, param), grad in zip(local_params.items(), grads)}
    return local_params
  # inner_update()
# MAMLNet