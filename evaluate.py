import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from FewShotEpisoder import FewShotEpisoder
from models.ResNetMAML import ResNetMAML
from torch import nn
import copy

def adapt(model, config, dataset, device, logging=False):
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
  from config import Config
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  my_data = torch.load("put your model path!!", map_location=device, weights_only=False)
  config, convig = my_data['config'], Config()
  my_model = ResNetMAML(config).to(device)
  my_model.load_state_dict(my_data["sate"])
  unseen_classes = [_ for _ in random.sample(list(convig.imageset.class_to_idx.values()), my_data["FRAMEWORK"]["n_way"])]
  evisoder = FewShotEpisoder(convig.imageset, unseen_classes, convig.k_shot, convig.n_query, config.transform, True)
  counts, n_problems = evaluate(my_model, evisoder=evisoder, config=config, device=device, logging=True)
  print(f"unseen classes: {evisoder.classes}\nACC: {(counts / n_problems):.2f}({counts}/{n_problems})")
# if __name__ == "__main__":