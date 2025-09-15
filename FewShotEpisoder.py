import random
import typing
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset

class FewShotDataset(Dataset):
  def __init__(self, dataset, classes: list, indices_t: list, transform: typing.Callable):
    self.dataset, self.indices_t, self.classes = dataset, indices_t, classes
    self.transform = transform
  # __init__()
  def __len__(self): return len(self.indices_t)

  def __getitem__(self, index: int):
    assert index < len(self.indices_t), IndexError("Index out of bounds") # check if index is out of bounds
    feature, label = self.dataset[self.indices_t[index]]
    feature = self.transform(feature)
    label = F.one_hot(torch.tensor(self.classes.index(label)), num_classes=len(self.classes)).float()
    return feature, label
  # __getitem__

  def _get_task_mean(self):
    whole_sum = list()
    for indice in self.indices_t: whole_sum.append(self.dataset[indice][0])
    whole_sum = torch.stack(whole_sum, dim=0)
    return torch.mean(whole_sum, dim=0)
  # _get_task_mean
# MAMLDataset

class FewShotEpisoder:
  def __init__(self, dataset, classes, k_shot, n_query, transform, is_val=False):
    assert k_shot > 0 or n_query > 0, ValueError("k_shot and n_query must be greater than 0.")
    self.dataset, self.classes = dataset, classes
    self.transform = transform
    self.k_shot, self.n_query = k_shot, n_query
    self.indices_t = self.get_indices_t()
    self.is_val = is_val
  # __init__()

  def get_indices_t(self):
    indices_t = {label: [] for label in range(self.classes.__len__())}
    for index, (_, label) in enumerate(self.dataset):
      if label in self.classes: indices_t[self.classes.index(label)].append(index)
    for label, _indices_t in indices_t.items():
      indices_t[label] = random.sample(_indices_t, self.k_shot + self.n_query)
    return indices_t
  # get_task_indices

  def get_episode(self):
    if self.is_val: return self.get_episode_val()
    tasks, query_set = list(), list()
    for indices_t in self.indices_t.values():
      indices_t = random.sample(indices_t, self.k_shot + self.n_query)
      tasks.append(FewShotDataset(self.dataset, self.classes, indices_t[:self.k_shot], self.transform))
      query_set.extend(indices_t[self.k_shot:])
    return tasks, FewShotDataset(self.dataset, self.classes, query_set, self.transform)

  def get_episode_val(self):
    support_set, query_set = list(), list()
    for indices_t in self.indices_t.values():
      indices_t = random.sample(indices_t, self.k_shot + self.n_query)
      support_set.extend(indices_t[:self.k_shot])
      query_set.extend(indices_t[self.k_shot:])
    return FewShotDataset(self.dataset, self.classes, support_set, self.transform), FewShotDataset(self.dataset, self.classes, query_set, self.transform)
# FewShotEpisoder