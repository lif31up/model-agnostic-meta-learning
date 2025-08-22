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

def train(DATASET, SAVE_TO, MODEL_CONFIG, TRAINING_CONFIG, device):
  # init a dataset
  imageset = tv.datasets.ImageFolder(root=DATASET)  # load dataset
  FRAMEWORK = {"n_way": 5, "k_shot": 5, "n_query": 2}
  seen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), FRAMEWORK["n_way"])]
  episoder = FewShotEpisoder(imageset, seen_classes, FRAMEWORK["k_shot"], FRAMEWORK["n_query"], transform)

  # initiate a model
  model = MAML(MODEL_CONFIG).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG["beta"])
  criterion = nn.CrossEntropyLoss()

  # meta training phase
  progress_bar, whole_loss = tqdm(range(TRAINING_CONFIG["epochs"])), 0.
  for _ in progress_bar:
    # inner loop: init local params, adapt to the task, ueses seen classes in support_set
    tasks, query_set = episoder.get_episode()
    local_params = list()
    for task in tasks: local_params.append(model.inner_update(task, TRAINING_CONFIG, device))
    # outer loop: update meta/global params, uses seen classes in query_set
    loss = 0.
    for feature, label in DataLoader(query_set, batch_size=TRAINING_CONFIG["epochs:batch_size"], shuffle=True,
                                     pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      task_loss = 0.
      for local_param in local_params:
        pred = model.forward(feature, local_param)
        task_loss += criterion(pred, label)
      loss += task_loss / len(local_params)  # calculate avg of losses per tasks
    # uses pytorch auto-grad to update global/meta params
    loss /= len(query_set)
    optim.zero_grad()
    loss.backward()
    optim.step()
    progress_bar.set_postfix(loss=loss.item())
    whole_loss += loss.item()
  # for

  # saving the model
  features = {
    "sate": model.state_dict(),
    "FRAMEWORK": FRAMEWORK,
    "MODEL_CONFIG": MODEL_CONFIG,
    "TRAINING_CONFIG": TRAINING_CONFIG,
  }  # feature
  torch.save(features, SAVE_TO + ".bin")
  save_file(model.state_dict(), SAVE_TO + ".safetensors")
# main

if __name__ == "__main__":
  from config import MODEL_CONFIG, TRAINING_CONFIG
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train(DATASET="./data/omniglot-py/images_background/Futurama", SAVE_TO="./5w1s", MODEL_CONFIG=MODEL_CONFIG, TRAINING_CONFIG=TRAINING_CONFIG, device=device)
# if __name__ == "__main__":