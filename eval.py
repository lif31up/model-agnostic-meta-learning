import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from FewShotEpisoder import FewShotEpisoder
from model.MAML import MAML
from torch import nn
from transform import *

def evaluate(MODEL_PATH, DATASET_PATH):
  # load a model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  data = torch.load(MODEL_PATH)["state"]
  state, MODEL_CONFIG, TRAINING_CONFIG, FRAMEWORK = data["state"], data["MODEL_CONFIG"], data["TRAINING_CONFIG"], data["FRAMEWORK"]
  model = MAML(MODEL_CONFIG).to(device)
  model.load_state_dict(state)
  model.eval()

  # create FewShotEpisoder which creates tuple of (support set, query set)
  imageset = tv.datasets.ImageFolder(root=DATASET_PATH)
  unseen_classes = random.sample(list(imageset.class_to_idx.values()), FRAMEWORK["n_way"])
  evisoder = FewShotEpisoder(imageset, unseen_classes, FRAMEWORK["k_shot"], FRAMEWORK["n_query"], transform)
  (_, query_set) = evisoder.get_episode()

  # FAST ADAPTION!!!
  optim = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG["alpha"])
  criterion = nn.CrossEntropyLoss()

  ITERFORADAPTION = 15
  progress_bar = tqdm(range(ITERFORADAPTION), desc="ADAPTING")
  for _ in progress_bar:
    for feature, label in DataLoader(query_set, batch_size=4, shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      pred = model.forward(feature)
      loss = criterion(pred, label)
      optim.zero_grad()
      loss.backward()
      optim.step()
  # for _ in progress_bar:

  # validate the model using
  count, n_problem = 0, len(query_set)
  for feature, label in DataLoader(query_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=4):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    pred = model.forward(feature)
    if torch.argmax(pred) == torch.argmax(label): count += 1
  # for feature, label in:
  print(
    f"seen classes: {data['seen_classes']}\nunseen classes: {unseen_classes}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
# evaluate

if __name__ == "__main__": evaluate("./model/5w1s.pth", "../data/omniglot-py/images_background/Futurama")