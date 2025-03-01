import random
import torch
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

  # overall configuration
  n_way, k_shot, n_query = data["framework"]

  # load imageset
  transform = data["transform"]  # define transform

  # create FewShotEpisoder which creates tuple of (support set, query set)
  imageset = tv.datasets.ImageFolder(root=DATASET)
  seen_classes, unseen_classes = data['seen_classes'], random.sample(list(imageset.class_to_idx.values()), n_way)
  episoder = FewShotEpisoder(imageset, unseen_classes, k_shot, n_query, transform)

  # fast adaption(inner loop)
  (tasks, query_set), adaptions = episoder.get_episode(), list()
  for task in tasks: adaptions.append(model.inner_update(task))

  # evaluate
  count, n_problem = 0, len(query_set)
  for feature, label in DataLoader(query_set, shuffle=True):
    task_i = torch.argmax(label).item()
    pred = model.forward(feature, adaptions[task_i])
    if torch.argmax(pred) == torch.argmax(label): count += 1
  # for
  print(f"seen classes: {seen_classes}\nunseen classes: {unseen_classes}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
# main

if __name__ == "__main__": evaluate("./model/model.pth", "../data/omniglot-py/images_background/Futurama")