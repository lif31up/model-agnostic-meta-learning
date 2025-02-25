import torch.optim
from torch import nn
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.MAMLDataset import MAMLEpisoder, MAMLDataset
import torch.nn.functional as F

def forward(x, params):
  x = F.conv2d(x, params['conv1.weight'], bias=params['conv1.bias'], stride=1, padding=1)
  x = F.relu(x)
  x = F.conv2d(x, params['conv2.weight'], bias=params['conv2.bias'], stride=1, padding=1)
  x = F.relu(x)
  x = x.flatten()
  x = F.linear(x, weight=params['l1.weight'], bias=params['l1.bias'])
  x = F.relu(x)
  return F.softmax(x)
# forward()

class MAML(nn.Module):
  def __init__(self, inpt_channels: int, hidn_channels: int, oupt_channels: int, lr: float):
    super(MAML, self).__init__()
    self.conv1 = nn.Conv2d(inpt_channels, hidn_channels, kernel_size=3, padding=1, stride=1)
    self.conv2 = nn.Conv2d(hidn_channels, hidn_channels, kernel_size=3, padding=1, stride=1)
    self.l1 = nn.Linear(in_features=200704, out_features=oupt_channels)
    self.relu, self.flatten, self.softmax = nn.ReLU(), nn.Flatten(), nn.Softmax()
    self.lr = lr
  # __init__

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.flatten(x)
    x = self.l1(x)
    x = self.relu(x)
    return self.softmax(x)
  # forward

  def inner_update(self, task: MAMLDataset):
    local_params = {name: param.clone() for name, param in self.named_parameters()}
    for feature, label in DataLoader(task, shuffle=True):
      pred = forward(feature, local_params)
      loss = nn.MSELoss()(pred, label)
      grads = torch.autograd.grad(loss, list(local_params.values()), create_graph=True)
      local_params = {name: param - self.lr * grad for (name, param), grad in zip(local_params.items(), grads)}
    return local_params
  # inner_update()
# MAMLNet

if __name__ == "__main__":
  tv.datasets.Omniglot(root="./data/", background=True, download=True)  # download dataset
  imageset = tv.datasets.ImageFolder(root="./data/omniglot-py/images_background/Futurama")  # load dataset

  # define transform
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])  # transform

  # 5 way 5 shot 설정을 적용합니다.
  n_way, k_shot, n_query = 5, 5, 2  # define framework to use

  # create FSL episode generator
  chosen_classes = list(imageset.class_to_idx.values())[:n_way]
  episoder = MAMLEpisoder(imageset, chosen_classes, k_shot, n_query, transform)

  # init model
  basic_config, meta_config = (0.001, 0.01), (0.001, 0.01)
  model = MAML((3, 3, n_way), basic_config, meta_config, 5)
  criterion = nn.MSELoss()
  optim = torch.optim.Adam(model.model.parameters(), lr=0.0001, weight_decay=0.01)

  # train loop
  iters, whole_loss = 50, float()
  for _ in tqdm(range(iters)):
    tasks, query_set = episoder.get_episode()
    fast_adaptions = list()
    for task in tasks:
      fast_adaptions.append(model.inner_update(task))
    loss = float()
    for feature, label in DataLoader(query_set, shuffle=True):
      for fast_adaption in fast_adaptions:
        pred = forward(feature, fast_adaption)
        loss += criterion(pred, label)
    # for
    loss /= n_way
    whole_loss += loss.item()
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss.item())
  # for
  print(whole_loss / iters)

  _, query_set = episoder.get_episode()
  cnt = 0
  for feature, label in DataLoader(query_set, shuffle=True):
    pred = model.model.forward(feature)
    if torch.argmax(pred) == torch.argmax(label): cnt += 1
  print(f"{cnt}/{len(query_set)}")
# if