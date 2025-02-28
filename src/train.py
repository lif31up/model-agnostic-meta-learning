import torch
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from src.model.MAML import MAML
from src.FewShotEpisoder import FewShotEpisoder
import random
from config import HYPERPARAMETER_CONFIG, TRAINING_CONFIG, MODEL_CONFIG

def train(DATASET: str, SAVE_TO: str, N_WAY: int, K_SHOT: int, N_QUERY: int):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # init device

  # define transform
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]) # transform

  # configuration
  iters, epochs = TRAINING_CONFIG["iters"], TRAINING_CONFIG["epochs"]
  alpha, beta = HYPERPARAMETER_CONFIG["alpha"], HYPERPARAMETER_CONFIG["beta"]
  model_config = (MODEL_CONFIG["n_inpt"], MODEL_CONFIG["n_hidn"], MODEL_CONFIG["n_oupt"], (iters, alpha))

  # create FSL episode generator
  imageset = tv.datasets.ImageFolder(root=DATASET)  # load dataset
  chosen_classes = random.sample(list(imageset.class_to_idx.values()), N_WAY)
  episoder = FewShotEpisoder(imageset, chosen_classes, K_SHOT, N_QUERY, transform)

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
        pred = model.forward(feature, fast_adaption)
        loss += criterion(pred, label)
    # for for

    # update global parameters
    loss /= N_WAY
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
    "model_config": model_config,
    "transform": transform,
    "chosen_classes": chosen_classes,
    "framework": (N_WAY, K_SHOT, N_QUERY)
  }  # features
  torch.save(features, SAVE_TO)
  print(f"model save to {SAVE_TO}")
# main

if __name__ == "__main__": train(DATASET="../data/omniglot-py/images_background/Futurama", SAVE_TO="./model/model.pth", N_WAY=5, K_SHOT=5, N_QUERY=2)
