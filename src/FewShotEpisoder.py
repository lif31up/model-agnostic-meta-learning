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
    """ Returns a sample from the dataset at the given index.
      Args: index of the sample to be retrieved.
      Returns: tuple of the transformed feature and the label. """
    assert index < len(self.indices_t), IndexError("Index out of bounds") # check if index is out of bounds
    feature, label = self.dataset[self.indices_t[index]]
    # apply transformation
    feature = self.transform(feature)
    label = F.one_hot(torch.tensor(self.classes.index(label)), num_classes=len(self.classes)).float()
    return feature, label
  # __getitem__()
# MAMLDataset

class FewShotEpisoder:
  def __init__(self, dataset, classes: list, k_shot: int, n_query: int, transform:typing.Callable):
    assert k_shot > 0 or n_query > 0, ValueError("k_shot and n_query must be greater than 0.")
    self.dataset, self.classes = dataset, classes
    self.transform = transform
    self.k_shot, self.n_query = k_shot, n_query
    self.indices_t = self.get_task_indices()
  # __init__()

  def get_task_indices(self):
    indices_t = {label: [] for label in self.classes}
    for index, (_, label) in enumerate(self.dataset):
      if label in self.classes: indices_t[label].append(index)
    return indices_t
  # get_task_indices

  def get_episode(self):
    tasks, query_set = list(), list()
    for indices_t in self.indices_t.values():
      indices_t = random.sample(indices_t, self.k_shot + self.n_query)
      tasks.append(FewShotDataset(self.dataset, self.classes, indices_t[:self.k_shot], self.transform))
      query_set.extend(indices_t[self.k_shot:])
    return tasks, FewShotDataset(self.dataset, self.classes, query_set, self.transform)
# MAMLDataset