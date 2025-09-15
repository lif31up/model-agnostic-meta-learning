import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from models.ResNet import ResNet_MAML
from FewShotEpisoder import FewShotEpisoder
import random

def train(model, path, config, episoder:FewShotEpisoder, device):
  assert isinstance(config, Config), "config is not a Config."

  model.to(device)
  optim = torch.optim.Adam(model.parameters(), lr=config.beta, eps=config.eps)
  criterion = nn.CrossEntropyLoss()

  progression = tqdm(range(config.epochs))
  for _ in progression:
    tasks, query_set = episoder.get_episode()
    local_params = list()
    for task in tasks: local_params.append(model.inner_update(task=task, device=device)) # inner loop: init local params, adapt to the task, ueses seen classes in support_set
    loss = float(0)
    for feature, label in DataLoader(query_set, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      task_loss = float(0)
      for local_param in local_params:
        pred = model.forward(feature, local_param)
        task_loss += criterion(pred, label)
      loss += task_loss / len(local_params) # calculate avg of losses per tasks
    loss /= query_set.__len__()
    optim.zero_grad()
    loss.backward()
    optim.step()
    progression.set_postfix(loss=loss.item())

  features = {
    "sate": model.state_dict(),
    "config": config
  } # feature
  torch.save(features, f"{path}.bin")
# train()

if __name__ == "__main__":
  from config import Config
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  maml_config = Config()
  imageset = maml_config.imageset
  seen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), maml_config.n_way)]
  episoder = FewShotEpisoder(imageset, seen_classes, maml_config.k_shot, maml_config.n_query, maml_config.transform)
  model = ResNet_MAML(maml_config)
  train(path=maml_config.save_to, model=model, config=maml_config, episoder=episoder, device=device)
# if __name__ == "__main__":