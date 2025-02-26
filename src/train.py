import torch
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from src.model.MAML import MAML
from src.MAMLDataset import MAMLEpisoder
import random

def main(path: str, save_to: str, n_way: int, k_shot: int, n_query: int, inner_iters: int, iters: int):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # init device

  imageset = tv.datasets.ImageFolder(root=path)  # load dataset

  # define transform
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]) # transform

  # set framework
  inner_lr, lr = 0.001, 0.001
  n_inpt, h_inpt, n_output = (3, 9, n_way)

  # create FSL episode generator
  chosen_classes = random.sample(list(imageset.class_to_idx.values()), n_way)
  episoder = MAMLEpisoder(imageset, chosen_classes, k_shot, n_query, transform)

  # init model
  model = MAML(n_inpt, h_inpt, n_output, inner_lr, inner_iters).to(device)
  criterion = nn.MSELoss()
  optim = torch.optim.Adam(model.parameters(), lr=lr)

  # train loop
  progress_bar, whole_loss = tqdm(range(iters)), float()

  for _ in progress_bar:
    tasks, query_set = episoder.get_episode()
    fast_adaptions = list()
    # 이너루프: 개발 작업에 연관되어 지역 매개변수를 갱신합니다.
    for task in tasks: fast_adaptions.append(model.inner_update(task, device))
    loss = float()
    # 아우터 루프: 모든 작업에 연관되어 메타/전역 매개변수를 갱신하여 이를 실제 모델에 반영합니다.
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
    "model config": (n_inpt, h_inpt, n_output),
    "episoder": episoder
  }  # features
  torch.save(features, save_to)
  print(f"model save to {save_to}")
# main

if __name__ == "__main__": main(path="../data/omniglot-py/images_background/Futurama", save_to="./model/model.pth", n_way=5, k_shot=5, n_query=2, inner_iters=20, iters=20)