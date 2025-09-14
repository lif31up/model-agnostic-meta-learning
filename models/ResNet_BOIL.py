import torch
from torch import nn
from torch.utils.data import DataLoader
from models.ResNet_MAML import ResNet_MAML

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