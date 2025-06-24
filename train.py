import torch
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from model.MAML import MAML
from FewShotEpisoder import FewShotEpisoder
import random
from safetensors.torch import save_file

def train(DATASET, SAVE_TO, config):
  transform = tv.transforms.Compose([
    tv.transforms.Resize((222, 222)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])  # transform

  # init a dataset
  imageset = tv.datasets.ImageFolder(root=DATASET)  # load dataset
  seen_classes = random.sample(list(imageset.class_to_idx.values()), config["n_way"])
  episoder = FewShotEpisoder(imageset, seen_classes, config["k_shot"], config["n_query"], transform)

  # initiate a model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = MAML(config).to(device=device)
  criterion, optim = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=config["beta"])

  # meta training phase
  progress_bar, whole_loss = tqdm(range(config["epochs"])), 0.
  for _ in progress_bar:
    # inner loop: initiate the local params and adapt them using `support set`, which consist if seen classes.
    tasks, query_set = episoder.get_episode()
    local_params = list()
    for task in tasks: local_params.append(model.inner_update(task))
    # outer loop: meta params update using `query set`, which consist of unseen classes.
    loss = 0.
    for feature, label in DataLoader(query_set, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      task_loss = 0.
      for local_param in local_params:
        pred = model.forward(feature, local_param)
        task_loss += criterion(pred, label)
      loss += task_loss / len(local_params)  # calculate the loss for the overall tasks
    # update the meta params using autograd
    loss /= len(query_set)
    optim.zero_grad()
    loss.backward()
    optim.step()
    progress_bar.set_postfix(loss=loss.item())
  # for

  # saving the model
  features = {
    "state": model.state_dict(),
    "config": config
  }  # feature
  torch.save(features, SAVE_TO + ".pth")
  save_file(model.state_dict(), SAVE_TO + ".safetensors")
# main

if __name__ == "__main__":
  from config import CONFIG

  train(DATASET="../data/omniglot-py/images_background/Futurama", SAVE_TO="./model/5w1s", config=CONFIG)
# if __name__ == "__main__":