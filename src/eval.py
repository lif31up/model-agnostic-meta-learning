import torch
from torch.utils.data import DataLoader
from src.model.MAML import MAML

def main(model: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load model
  data = torch.load(model)
  state = data["state"]
  episoder = data["episoder"]
  model = MAML(*data["model config"]).to(device)
  model.load_state_dict(state)
  model.eval()

  tasks, query_set = episoder.get_episode()
  cnt = 0
  for feature, label in DataLoader(query_set, shuffle=True):
    pred = model.forward(feature)
    if torch.argmax(pred) == torch.argmax(label): cnt += 1
  # for
  print(f"accuracy: {cnt / len(query_set):.2f}({cnt}/{len(query_set)})")
# main

if __name__ == "__main__": main(model="./model/model.pth")