This implementation is inspired by [**"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"**](https://arxiv.org/abs/1703.05175) (2017) by Jake Snell, Kevin Swersky, Richard S. Zemel.
* **Note & Reference:** [GitBook](https://lif31up.gitbook.io/lif31up/few-shot-learning/model-agnostic-meta-learning-for-fast-adaptation-of-deep-networks)
* **Quickstart on Colab:** [Colab]()

|            | 5 Way ACC (5 shot) | 5 Way ACC(1 shot) |
|------------|--------------------|------------------|
|**Omniglot**|`100%` **(100/100)**|`96%` **(96/100)**|

## Model-Agnostic Meta-Learning for Few-Shot Image Classification
This repository implements a Model-Agnostic Meta-Learning (MAML) algorithm for few-shot image classification tasks using PyTorch.

* **Task**: classifying image with few dataset.
* **Dataset**: `omniglot futurama`

Few-shot learning focuses on enabling models to generalize to new tasks with only a few labeled examples. MAML achieves this by optimizing for a set of parameters that can quickly adapt to new tasks through gradient-based updates, allowing the model to efficiently learn from limited data.

* **Inner-Loop Fast Adaption:** For each task, the model is fine-tuned on a small support set using a few gradient steps to minimize task-specific loss.
* **Meta-Update (Outer Loop):** After task-specific updates, gradients are computed based on query set performance, and the initial model parameters are updated to improve adaptability across tasks.

---
### Configuration
confing.py contains the configuration settings for the model, including the framework, dimensions, learning rate, and other hyperparameters
```python
CONFIG = {
  "version": "1.0.1",
  # framework
  "n_way": 5,
  "k_shot": 1,
  "n_query": 2,
  # model
  "inpt_dim": 3,
  "hidn_dim": 6,
  "oupt_dim": 5,
  # hp
  "iters": 5,
  "epochs": 10,
  "batch_size": 8,
  "inner_batch_size": 5,
  "alpha": 1e-2,
  "beta": 1e-4,
} # CONFIG
```
### Training
train.py is a script to train the model on the omniglot dataset. It includes the training loop, evaluation, and saving the model checkpoints.
```python
if __name__ == "__main__":
  from config import CONFIG

  train(DATASET="../data/omniglot-py/images_background/Futurama", SAVE_TO="./model/5w1s", config=CONFIG)
# if __name__ == "__main__":
```
### Evaluation
eval.py is used to evaluate the trained model on the omniglot dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.
```python
if __name__ == "__main__": evaluate("./model/5w1s.pth", "../data/omniglot-py/images_background/Futurama")
# output example:
# seen classes: [1, 15, 6, 20, 12]
# unseen classes: [22, 3, 16, 20, 18]
# accuracy: 0.9000(9/10)
```
---
## Technical Highlights

### Inner Loop
```python
def inner_update(self, task):
  local_params = {name: param.clone() for name, param in self.named_parameters()}
  for _ in range(self.epochs):
    for feature, label in DataLoader(task, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True):
      feature, label = feature.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)
      pred = self.forward(feature, local_params)
      loss = nn.MSELoss()(pred, label)
      grads = torch.autograd.grad(loss, list(local_params.values()), create_graph=True)
      local_params = {name: param - (self.alpha * grad) for (name, param), grad in zip(local_params.items(), grads)}
  # for for
  return local_params
# inner_update()
```
### Outer Loop


```python
tasks, query_set = episoder.get_episode()
local_params = list()
for task in tasks: local_params.append(maml.inner_update(task))
for feature, label in DataLoader(query_set, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
  feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
  for local_param in local_params:
    pred = maml.forward(feature, local_param)
    print(f"pred shape: {pred.shape} feature shape: {feature.shape} label shape: {label.shape}")
  # for
  break
# for
```

### Forward
```python
def forward(self, x, params=None):
  if not params: params = dict(self.named_parameters())
  x = F.conv2d(x, params['conv1.weight'], bias=params['conv1.bias'], stride=1, padding=1)
  x = self.swish(x)
  x = F.conv2d(x, params['conv2.weight'], bias=params['conv2.bias'], stride=1, padding=1)
  x = self.pool(x)
  x = self.flatten(x)
  x = F.linear(x, weight=params['l1.weight'], bias=params['l1.bias'])
  return self.softmax(x)
# forward()
```