import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision as tv
from FewShotEpisoder import FewShotEpisoder
from model.MAML import MAML

def evaluate(MODEL: str, DATASET: str):
  # load a model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  data = torch.load(MODEL)["state"]
  state, config = data["state"], data["config"]
  model = MAML(config).to(device)
  model.load_state_dict(state)
  model.eval()

  # define an encoder
  transform = tv.transforms.Compose([
    tv.transforms.Resize((222, 222)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]) # transform

  # create FewShotEpisoder which creates tuple of (support set, query set)
  imageset = tv.datasets.ImageFolder(root=DATASET)
  unseen_classes = random.sample(list(imageset.class_to_idx.values()), config["n_way"])
  evisoder = FewShotEpisoder(imageset, unseen_classes, config["k_shot"], config["n_query"], transform)

  # fast adaption using inner loop
  accuracy, count, n_problem = 0., 0, 0
  (tasks, query_set), adaptions = evisoder.get_episode(), list()
  for task in tqdm(tasks, desc="adaption"): adaptions.append(model.inner_update(task, device))

  # meta validation phase
  for feature, label in DataLoader(query_set, shuffle=True, num_workers=4, pin_memory=True):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    task_i = torch.argmax(label).item()
    pred = model.forward(feature, adaptions[task_i])
    if torch.argmax(pred) == torch.argmax(label): count += 1
    n_problem += 1
  # for

  print(f"seen classes: {data['seen_classes']}\nunseen classes: {unseen_classes}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
# evaluate

if __name__ == "__main__": evaluate("./model/5w1s.pth", "../data/omniglot-py/images_background/Futurama")