import torch
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from src.model.MAML import MAML
from src.MAMLEpisoder import MAMLEpisoder
import random

def main(path: str, save_to: str, n_way: int, k_shot: int, n_query: int, iters: int, epochs: int):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # init device

  imageset = tv.datasets.ImageFolder(root=path)  # load dataset

  # define transform
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]) # transform

  # set model config and hps
  alpha, beta = 0.001, 0.001
  model_config = (3, 9, n_way, (iters, alpha))

  # create FSL episode generator
  chosen_classes = random.sample(list(imageset.class_to_idx.values()), n_way)
  episoder = MAMLEpisoder(imageset, chosen_classes, k_shot, n_query, transform)

  # init model
  model = MAML(*model_config).to(device)
  criterion = nn.MSELoss()
  optim = torch.optim.Adam(model.parameters(), lr=beta)

  # train loop
  progress_bar, whole_loss = tqdm(range(epochs)), float()
  for _ in progress_bar:
    tasks, query_set = episoder.get_episode()
    fast_adaptions = list()
    # inner loop
    for task in tasks: fast_adaptions.append(model.inner_update(task, device))
    loss = float()
    # outer loop
    for feature, label in DataLoader(query_set, shuffle=True):
      for fast_adaption in fast_adaptions:
        pred = model._forward(feature, fast_adaption)
        loss += criterion(pred, label)
    # for for

    # update global parameters
    loss /= n_way
    optim.zero_grad()
    loss.backward()
    optim.step()

    # print loss
    progress_bar.set_postfix(loss=loss.item())
    whole_loss += loss.item()
  # for
  print(f"train ended with whole loss: {whole_loss / iters:.4f}")

  # saving the model's parameters and the other data
  features = {
    "state": model.state_dict(),
    "transform": transform,
    "model config": model_config,
    "episoder": episoder
  }  # features
  torch.save(features, save_to)
  print(f"model save to {save_to}")
# main

if __name__ == "__main__": main(path="../data/omniglot-py/images_background/Futurama", save_to="./model/model.pth", n_way=5, k_shot=5, n_query=2, epochs=10, iters=20)
