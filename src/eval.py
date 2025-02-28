import random
from tqdm import tqdm
from config import HYPERPARAMETER_CONFIG
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision as tv
from src.FewShotEpisoder import FewShotEpisoder
from src.model.MAML import MAML

def evaluate(MODEL: str, DATASET: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load model
  data = torch.load(MODEL)
  model = MAML(*data["model_config"]).to(device)
  model.load_state_dict(data["state"])

  # create FSL episode generator
  n_way, k_shot, n_query = data["framework"]
  imageset = tv.datasets.ImageFolder(root=DATASET)  # load dataset
  unseen_classes = random.sample(list(imageset.class_to_idx.values()), n_way)
  episoder = FewShotEpisoder(imageset, unseen_classes, k_shot, n_query, data["transform"])

  progress_bar, whole_loss = tqdm(range(5)), 0.
  criterion, optim = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=HYPERPARAMETER_CONFIG["beta"])
  tasks, query_set = episoder.get_episode()
  for _ in progress_bar:
    fast_adaptions = list()
    for task in tasks: fast_adaptions.append(model.inner_update(task, device))  # inner loop
    loss = float()
    for feature, label in DataLoader(query_set, shuffle=True):  # outer loop
      task_loss = 0.
      for fast_adaption in fast_adaptions:
        pred = model.forward(feature, fast_adaption)
        task_loss += criterion(pred, label)  # sum loss of each tasks
      loss += task_loss / len(fast_adaptions)  # sum / number of tasks
    # update params using autograd
    loss /= len(query_set)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # progressing and whole loss
    progress_bar.set_postfix(loss=loss.item())
    whole_loss += loss.item()
  # adaption

  model.eval()
  cnt, n_query_set = 0, len(query_set)
  for feature, label in DataLoader(query_set, shuffle=True):
    pred = model.forward(feature)
    if torch.argmax(pred) == torch.argmax(label): cnt += 1
  # for
  print(f"accuracy: {cnt / n_query_set:.2f}({cnt}/{n_query_set})")
# main

if __name__ == "__main__": evaluate("./model/model.pth", "../data/omniglot-py/images_background/Futurama")