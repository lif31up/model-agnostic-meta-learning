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

  cnt, iters, n_query_set = 0, 5, episoder.n_query * len(episoder.classes)
  for _ in range(iters):
    _, query_set = episoder.get_episode()
    for feature, label in DataLoader(query_set, shuffle=True):
      pred = model.forward(feature)
      if torch.argmax(pred) == torch.argmax(label): cnt += 1
    # for
  print(f"accuracy: {cnt / n_query_set * iters:.2f}({cnt}/{n_query_set * iters})")
# main

if __name__ == "__main__": main(model="./model/model.pth")