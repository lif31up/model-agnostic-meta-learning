import torch
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from model.MAML import MAML
from FewShotEpisoder import FewShotEpisoder
import random
from safetensors.torch import save_file
from transform import *

def train(model, path, config, episoder, device):
  model.to(device)
  optim = torch.optim.Adam(model.parameters(), lr=config["beta"])
  criterion = nn.CrossEntropyLoss()

  progress_bar, whole_loss = tqdm(range(config["epochs"])), 0.
  for _ in progress_bar:
    # inner loop: init local params, adapt to the task, ueses seen classes in support_set
    tasks, query_set = episoder.get_episode()
    local_params = list()
    for task in tasks: local_params.append(model.inner_update(task, config, device))
    # outer loop: update meta/global params, uses seen classes in query_set
    loss = 0.
    for feature, label in DataLoader(query_set, batch_size=config["epochs:batch_size"], shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      task_loss = 0.
      for local_param in local_params:
        pred = model.forward(feature, local_param)
        task_loss += criterion(pred, label)
      loss += task_loss / len(local_params) # calculate avg of losses per tasks
    # uses pytorch auto-grad to update global/meta params
    loss /= len(query_set)
    optim.zero_grad()
    loss.backward()
    optim.step()
    progress_bar.set_postfix(loss=loss.item())
    whole_loss += loss.item()
  # for _ in progress_bar

  features = {
    "sate": model.state_dict(),
    "FRAMEWORK": FRAMEWORK,
    "MODEL_CONFIG": MODEL_CONFIG,
    "TRAINING_CONFIG": TRAINING_CONFIG
  } # feature
  torch.save(features, PATH)
  return 0
# train()

if __name__ == "__main__":
  from config import MODEL_CONFIG, TRAINING_CONFIG, FRAMEWORK
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tv.datasets.Omniglot(root="./data/", background=True, download=True)
  imageset = tv.datasets.ImageFolder(root="./data/omniglot-py/images_background/Futurama")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  PATH = "/content/drive/MyDrive/Colab Notebooks/MAML.bin"
  seen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), FRAMEWORK["n_way"])]
  episoder = FewShotEpisoder(imageset, seen_classes, FRAMEWORK["k_shot"], FRAMEWORK["n_query"], transform)
  model = MAML(MODEL_CONFIG)
  train(path=PATH, model=model, config=TRAINING_CONFIG, episoder=episoder, device=device)
# if __name__ == "__main__":