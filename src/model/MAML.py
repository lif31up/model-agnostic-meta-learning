import torch.optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

class MAML(nn.Module):
  def __init__(self, config):
    super(MAML, self).__init__()
    self.conv1 = nn.Conv2d(config["inpt_dim"], config["hidn_dim"], kernel_size=3, padding=1, stride=1)
    self.conv2 = nn.Conv2d(config["hidn_dim"], config["hidn_dim"], kernel_size=3, padding=1, stride=1)
    self.l1 = nn.Linear(in_features=32856, out_features=config["oupt_dim"])
    self.swish, self.flatten, self.softmax, self.pool = nn.SiLU(), nn.Flatten(1), nn.Softmax(dim=1), nn.MaxPool2d(3)
    self.epochs, self.alpha, self.batch_size = config["epochs"], config["alpha"], config["inner_batch_size"]
  # __init__()

  def to(self, *args, **kwargs):
    super(MAML, self).to(*args, **kwargs)
    self.device = kwargs["device"]
    return self
  # to()

  def forward(self, x, params=None):
    if not params: params = dict(self.named_parameters())
    x = F.conv2d(x, params['conv1.weight'], bias=params['conv1.bias'], stride=1, padding=1)
    x = self.swish(x)
    x = F.conv2d(x, params['conv2.weight'], bias=params['conv2.bias'], stride=1, padding=1)
    x = self.pool(x)
    x = self.flatten(x)
    x = F.linear(x, weight=params['l1.weight'], bias=params['l1.bias'])
    return self.softmax(x)
  # forward()

  def inner_update(self, task):
    local_params = {name: param.clone() for name, param in self.named_parameters()}
    for _ in range(self.epochs):
      for feature, label in DataLoader(task, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True):
        feature, label = feature.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)
        pred = self.forward(feature, local_params)
        loss = nn.MSELoss()(pred, label)
        grads = torch.autograd.grad(loss, list(local_params.values()), create_graph=True)
        local_params = {name: param - (self.alpha * grad) for (name, param), grad in zip(local_params.items(), grads)}
    # for for
    return local_params
  # inner_update()
# MAMLNet

if __name__ == "__main__":
  from config import CONFIG
  import torchvision as tv
  from src.FewShotEpisoder import FewShotEpisoder

  transform = tv.transforms.Compose([
    tv.transforms.Resize((222, 222)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]) # transform

  imageset = tv.datasets.ImageFolder(root="../../data/omniglot-py/images_background/Futurama")  # load dataset
  seen_classes = random.sample(list(imageset.class_to_idx.values()), CONFIG["n_way"])
  episoder = FewShotEpisoder(imageset, seen_classes, CONFIG["k_shot"], CONFIG["n_query"], transform)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  maml = MAML(config=CONFIG)
  maml.to(device=device)

  tasks, query_set = episoder.get_episode()
  local_params = list()
  for task in tasks: local_params.append(maml.inner_update(task))
  for feature, label in DataLoader(query_set, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    for local_param in local_params:
      pred = maml.forward(feature, local_param)
      print(f"pred shape: {pred.shape} feature shape: {feature.shape} label shape: {label.shape}")
    # for
    break
  # for
# if __name__ == "__main__":