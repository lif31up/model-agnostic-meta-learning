import torch
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from src.model.MAML import MAML
from src.FewShotEpisoder import FewShotEpisoder
import random
from config import HYPER_PARAMETERS, TRAINING_CONFIG, MODEL_CONFIG, FRAMEWORK
from safetensors.torch import save_file

def train(DATASET: str, SAVE_TO: str):
  # overall configuration
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_way, k_shot, n_query = FRAMEWORK["n_way"], FRAMEWORK["k_shot"], FRAMEWORK["n_query"]
  iters, epochs, batch_size = TRAINING_CONFIG["iters"], TRAINING_CONFIG["epochs"], TRAINING_CONFIG["batch_size"]
  alpha, beta = HYPER_PARAMETERS["alpha"], HYPER_PARAMETERS["beta"]
  model_config = (MODEL_CONFIG["in_channels"], MODEL_CONFIG["hidden_channels"], MODEL_CONFIG["output_channels"], (iters, alpha))

  # define transform
  transform = tv.transforms.Compose([
    tv.transforms.Resize((222, 222)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])  # transform

  # creat episoder for few-shot learning
  imageset = tv.datasets.ImageFolder(root=DATASET)  # load dataset
  seen_classes = random.sample(list(imageset.class_to_idx.values()), n_way)
  episoder = FewShotEpisoder(imageset, seen_classes, k_shot, n_query, transform)

  # initiate model
  model = MAML(*model_config).to(device)
  criterion, optim = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=beta)

  # META TRAINING PHASE
  progress_bar, whole_loss = tqdm(range(epochs)), 0.
  for _ in progress_bar:
    tasks, query_set = episoder.get_episode()
    # inner loop: initiate local params and adapt them using `support set` which is seen class.
    local_params = list()
    for task in tasks: local_params.append(model.inner_update(task, device))
    # outer loop: meta params update using `query set` which is unseen class.
    loss = 0.
    for feature, label in DataLoader(query_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      task_loss = 0.
      for local_param in local_params:
        pred = model.forward(feature, local_param)
        task_loss += criterion(pred, label)
      loss += task_loss / len(local_params)  # calculate loss for tasks
    # update meta params using autograd
    loss /= len(query_set)
    optim.zero_grad()
    loss.backward()
    optim.step()
    progress_bar.set_postfix(loss=loss.item())

  # saving model
  features = {
    "state": model.state_dict(),
    "FRAMEWORK": FRAMEWORK,
    "MODEL_CONFIG": MODEL_CONFIG,
    "HYPER_PARAMETERS": HYPER_PARAMETERS,
    "TRAINING_CONFIG": TRAINING_CONFIG,
    "TRANSFORM": transform,
    "seen_classes": seen_classes
  }  # feature
  torch.save(features, SAVE_TO)
  save_file(model.state_dict(), SAVE_TO.replace(".pth", ".safetensors"))
# main

if __name__ == "__main__": train(DATASET="../data/omniglot-py/images_background/Futurama", SAVE_TO="./model/5w1s.pth")
