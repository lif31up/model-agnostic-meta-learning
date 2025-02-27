import torch
from torch.utils.data import DataLoader
import torchvision as tv
from src.FewShotEpisoder import FewShotEpisoder
from src.model.MAML import MAML

def evaluate(MODEL: str, DATASET: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load model
  data = torch.load(MODEL)
  state = data["state"]
  model = MAML(*data["model_config"]).to(device)
  model.load_state_dict(state)
  model.eval()

  # create FSL episode generator
  imageset = tv.datasets.ImageFolder(root=DATASET)  # load dataset
  episoder = FewShotEpisoder(imageset, data["chosen_classes"], 1, 10, data["transform"])

  _, query_set = episoder.get_episode()
  cnt, n_query_set = 0, episoder.n_query * len(episoder.classes)
  for feature, label in DataLoader(query_set, shuffle=True):
    pred = model.forward(feature)
    if torch.argmax(pred) == torch.argmax(label): cnt += 1
  # for
  print(f"accuracy: {cnt / n_query_set:.2f}({cnt}/{n_query_set})")
# main

if __name__ == "__main__": evaluate("./model/model.pth", "../data/omniglot-py/images_background/Futurama")