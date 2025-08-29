import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from FewShotEpisoder import FewShotEpisoder
from model.MAML import MAML
from torch import nn
from transform import *
import copy

def adapt(model, config, dataset, device, logging=False ):
  model = copy.deepcopy(model).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=config["beta"])
  criterion = nn.CrossEntropyLoss()

  progress_bar = range(config["iterations"])
  if logging: progress_bar = tqdm(progress_bar, desc="ADAPTING", leave=True)
  for _ in progress_bar:
    for feature, label in DataLoader(dataset, batch_size=config["iterations:batch_size"], shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      pred = model.forward(feature)
      loss = criterion(pred, label)
      optim.zero_grad()
      loss.backward()
      optim.step()
    if logging: progress_bar.set_postfix(loss=loss.item())
  return model
# adapt

def evaluate(model, evisoder, config, device, logging=False):
  assert evisoder.is_val, "episoder.is_val should be True."

  (dataset, testset) = evisoder.get_episode_val()

  adapted_model = adapt(model=model, dataset=dataset, config=config, device=device, logging=logging)
  adapted_model.eval()
  counts, n_problems = 0, len(testset)
  for feature, label in DataLoader(testset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    pred = adapted_model.forward(feature)
    if torch.argmax(pred) == torch.argmax(label): counts += 1
  return counts, n_problems
# validation

if __name__ == "__main__":
  tv.datasets.Omniglot(root="./data/", background=True, download=True)
  imageset = tv.datasets.ImageFolder(root="./data/omniglot-py/images_background/Futurama")

  VAL_CONFIG = {
    "iterations": 100,
    "beta": 1e-4,
    "iterations:batch_size": 32,
  }  # VALIDATION_CONFIG
  VAL_FRAMEWORK = {"n_way": 5, "k_shot": 3, "n_query": 10}
  print(f"Validated Framework: {VAL_FRAMEWORK}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  my_data = torch.load("/content/drive/MyDrive/Colab Notebooks/MAML.bin", map_location=device, weights_only=False)
  my_model = MAML(my_data["MODEL_CONFIG"]).to(device)
  my_model.load_state_dict(my_data["sate"])

  unseen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), my_data["FRAMEWORK"]["n_way"])]
  evisoder = FewShotEpisoder(imageset, unseen_classes, VAL_FRAMEWORK["k_shot"], VAL_FRAMEWORK["n_query"], transform, True)
  counts, n_problems = evaluate(my_model, evisoder=evisoder, config=VAL_CONFIG, device=device, logging=True)
  print(f"unseen classes: {evisoder.classes}\nACC: {(counts / n_problems):.2f}({counts}/{n_problems})")
# if __name__ == "__main__":