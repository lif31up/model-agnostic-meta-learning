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
  seen_classes = data['seen_classes']
  unseen_classes = random.sample(list(imageset.class_to_idx.values()), n_way)
  episoder = FewShotEpisoder(imageset, unseen_classes, k_shot, n_query, data["transform"])

  tasks, query_set = episoder.get_episode()
  adaptions = list()
  for task in tasks: adaptions.append(model.inner_update(task))

  count, n_problem = 0, len(query_set)
  for feature, label in DataLoader(query_set, shuffle=True):
    task_i = torch.argmax(label).item()
    pred = model.forward(feature, adaptions[task_i])
    if torch.argmax(pred) == torch.argmax(label): count += 1
  # for
  print(f"seen classes: {seen_classes}\nunseen classes: {unseen_classes}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
  # main()
# main

if __name__ == "__main__": evaluate("./model/model.pth", "../data/omniglot-py/images_background/Futurama")