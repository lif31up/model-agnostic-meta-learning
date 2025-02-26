import torch
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from src.model.MAML import MAML, forward
from src.MAMLDataset import MAMLEpisoder

def main(path: str, save_to: str, iters: int = 50, n_way: int = 5, k_shot: int = 5, n_query: int= 2):
  tv.datasets.Omniglot(root=path, background=True, download=True)  # download dataset
  imageset = tv.datasets.ImageFolder(root="./data/omniglot-py/images_background/Futurama")  # load dataset

  # define transform
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])  # transform

  # create FSL episode generator
  chosen_classes = list(imageset.class_to_idx.values())[:n_way]
  episoder = MAMLEpisoder(imageset, chosen_classes, k_shot, n_query, transform)

  # init model
  model_config = (3, 4, n_way, 0.001)
  model = MAML(*model_config)
  criterion = nn.MSELoss()
  optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

  # train loop
  whole_loss = 50
  for _ in tqdm(range(iters)):
    tasks, query_set = episoder.get_episode()
    fast_adaptions = list()
    for task in tasks:
      fast_adaptions.append(model.inner_update(task))
    loss = float()
    for feature, label in DataLoader(query_set, shuffle=True):
      for fast_adaption in fast_adaptions:
        pred = forward(feature, fast_adaption)
        loss += criterion(pred, label)
    # for for
    loss /= n_way
    whole_loss += loss.item()
    optim.zero_grad()
    loss.backward()
    optim.step()
  # for
  print(f"whole loss: {whole_loss / iters:.4f}")

  # saving the model's parameters and the other data
  features = {
    "state": model.state_dict(),
    "episoder": episoder,
    "model config": model_config,
  }  # features
  torch.save(features, save_to)
# main()

if __name__ == "__main__": main(path="./data/", save_to="./model/model.pth", iters=50, n_way=5, k_shot=5, n_query=2)