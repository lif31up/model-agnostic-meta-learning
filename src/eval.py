import torch

from src.model.MAML import MAML


def main(model: str = "./model/model.pth"):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load model
  data = torch.load(model)
  state = data["state"]
  transform = data["transform"]
  model = MAML().to(device)
  model.load_state_dict(state)
  model.eval()

  # create FSL episode generator
  imageset = tv.datasets.ImageFolder(root=path)
  chosen_classes = list(imageset.class_to_idx.values())[:n_way]