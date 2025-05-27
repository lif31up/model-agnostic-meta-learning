import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision as tv
from src.FewShotEpisoder import FewShotEpisoder
from src.model.MAML import MAML

def evaluate(MODEL: str, DATASET: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load model
  data = torch.load(MODEL)
  model_config = (
  data["MODEL_CONFIG"]["in_channels"], data["MODEL_CONFIG"]["hidden_channels"], data["MODEL_CONFIG"]["output_channels"],
  (15, data["HYPER_PARAMETERS"]["alpha"]))
  model = MAML(*model_config).to(device)
  model.load_state_dict(data["sate"])

  # overall configuration
  n_way, k_shot, n_query = data["FRAMEWORK"].values()
  transform = data["TRANSFORM"]

  # evaluate
  accuracy, count, n_problem = 0., 0, 0

  # create FewShotEpisoder which creates tuple of (support set, query set)
  imageset = tv.datasets.ImageFolder(root=DATASET)
  unseen_classes = random.sample(list(imageset.class_to_idx.values()), n_way)
  evisoder = FewShotEpisoder(imageset, unseen_classes, k_shot, n_query, transform)

  # fast adaption(inner loop)
  (tasks, query_set), adaptions = evisoder.get_episode(), list()
  for task in tqdm(tasks, desc="adaption"): adaptions.append(model.inner_update(task, device))

  # evaluate
  for feature, label in DataLoader(query_set, shuffle=True):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    task_i = torch.argmax(label).item()
    pred = model.forward(feature, adaptions[task_i])
    if torch.argmax(pred) == torch.argmax(label): count += 1
    n_problem += 1
  # for

  print(f"seen classes: {data['seen_classes']}\nunseen classes: {unseen_classes}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
# evaluate

def evaluates(MODEL: str, DATASET: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"the device type of {device.type}.")

  # load model
  data = torch.load(MODEL)
  model_config = (data["MODEL_CONFIG"]["in_channels"], data["MODEL_CONFIG"]["hidden_channels"], data["MODEL_CONFIG"]["output_channels"], (15, data["HYPER_PARAMETERS"]["alpha"]))
  model = MAML(*model_config).to(device)
  model.load_state_dict(data["sate"])

  # overall configuration
  n_way, k_shot, n_query = data["FRAMEWORK"].values()
  transform = data["TRANSFORM"]

  # evaluate
  accuracy, n_eval = 0., 10
  count, n_problem, progress_bar = 0, 0, tqdm(range(n_eval), desc="evaluation")
  for _ in progress_bar:
    # create FewShotEpisoder which creates tuple of (support set, query set)
    imageset = tv.datasets.ImageFolder(root=DATASET)
    unseen_classes = random.sample(list(imageset.class_to_idx.values()), n_way)
    evisoder = FewShotEpisoder(imageset, unseen_classes, k_shot, n_query, transform)

    # fast adaption(inner loop)
    (tasks, query_set), adaptions = evisoder.get_episode(), list()
    for task in tasks: adaptions.append(model.inner_update(task, device))

    # evaluate
    for feature, label in DataLoader(query_set, shuffle=True):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      task_i = torch.argmax(label).item()
      pred = model.forward(feature, adaptions[task_i])
      if torch.argmax(pred) == torch.argmax(label): count += 1
      n_problem += 1
    # for
    accuracy += count / n_problem
    progress_bar.set_postfix(accuracy=count / n_problem)
  print(f"seen classes: {data['seen_classes']}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
# main

if __name__ == "__main__": evaluate("./model/5w1s.pth", "../data/omniglot-py/images_background/Futurama")