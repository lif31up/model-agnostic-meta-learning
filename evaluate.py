import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from FewShotEpisoder import FewShotEpisoder
from models.ResNet import ResNet_BOIL
from torch import nn
import copy

def adapt(model, config, dataset, device, logging=False):
  model = copy.deepcopy(model).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=config.alpha)
  criterion = nn.CrossEntropyLoss()
  progress_bar = range(config.iterations)
  if logging: progress_bar = tqdm(progress_bar, desc="ADAPTING", leave=True)
  for _ in progress_bar:
    loss = float(0)
    for feature, label in DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4):
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
# evaluate

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  my_data = torch.load(f='/content/drive/MyDrive/Colab Notebooks/ResNet_BOIL_.bin', map_location="cpu", weights_only=False)
  my_config = my_data['config']
  my_model = ResNet_BOIL(my_config)
  my_model.load_state_dict(my_data['sate'])
  imageset = my_config.imageset
  unseen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), my_config.n_way)]
  evisoder = FewShotEpisoder(imageset, unseen_classes, 5, 10, my_config.transform, is_val=True)
  n_counts, n_problems = evaluate(model=my_model, evisoder=evisoder, config=my_config, device=device, logging=True)
  print(f'{n_counts / n_problems:.2f}({n_counts}/{n_problems})')
# if __name__ == "__main__":