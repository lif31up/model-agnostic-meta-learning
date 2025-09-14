This implementation is inspired by [**"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"**](https://arxiv.org/abs/1703.05175) (2017) by Jake Snell, Kevin Swersky, Richard S. Zemel.
* **Note & Reference:** [GitBook](https://lif31up.gitbook.io/lif31up/few-shot-learning/model-agnostic-meta-learning-for-fast-adaptation-of-deep-networks)
* **⭐ Colab Playground:** [Colab](https://colab.research.google.com/drive/1ZmtP8rMZsSN_yA6tz3IKQU0ECXeAI018?usp=sharing)
* **Huggingface:** [Huggingface](https://huggingface.co/lif31up/model-agnostic-meta-learning)

|            | 5 Way ACC (1 shot) | 5 Way ACC(5 shot) |
|------------|-------------------|-------------------|
|**Omniglot**| `76%` **(76/50)** | `86%` **(86/100)** |

## Model-Agnostic Meta-Learning for Few-Shot Image Classification
The main purpose was to implement the from-scratch Model-Agnostic Meta-Learning (MAML) algorithm that's easy to execute on educational cloud environments.

![img_1.png](img_0.png)

* **Task**: classifying image with few dataset.
* **Dataset**: `omniglot futurama`

FSL(Few-Shot Learning) focuses on enabling models to generalize to new tasks with only a few labeled examples. 
MAML achieves this by optimizing for a set of parameters that can quickly adapt to new tasks through gradient-based updates, allowing the model to efficiently learn from limited data.

---
### Configuration
confing.py contains the configuration settings for the model, including the framework, dimensions, learning rate, and other hyperparameters

```python
MODEL_CONFIG = {
  "input_channels": 1,
  "hidden_channels": 32,
  "output_channels": 5,
  "conv:kernel_size": 3,
  "conv:padding": 1,
  "conv:stride": 1,
  "l1_in_features": 2592
} # MODEL_CONFIG

TRAINING_CONFIG = {
  "iterations": 100,
  "epochs": 30,
  "alpha": 1e-3,
  "beta": 1e-4,
  "iterations:batch_size": 32,
  "epochs:batch_size": 32,
} # TRAINING_CONFIG

FRAMEWORK = { "n_way": 5, "k_shot": 1, "n_query": 2 }
```
### Training
train.py is a script to train the model on the omniglot dataset. It includes the training loop, evaluation, and saving the model checkpoints.
```python
if __name__ == "__main__":
  from config import MODEL_CONFIG, TRAINING_CONFIG, FRAMEWORK
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tv.datasets.Omniglot(root="./data/", background=True, download=True)
  imageset = tv.datasets.ImageFolder(root="./data/omniglot-py/images_background/Futurama")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  PATH = "5w.bin"
  seen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), FRAMEWORK["n_way"])]
  episoder = FewShotEpisoder(imageset, seen_classes, FRAMEWORK["k_shot"], FRAMEWORK["n_query"], transform)
  model = MAML(MODEL_CONFIG)
  train(path=PATH, model=model, config=TRAINING_CONFIG, episoder=episoder, device=device)
# if __name__ == "__main__":
```
### Evaluation
eval.py is used to evaluate the trained model on the omniglot dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.
```python
if __name__ == "__main__":
  tv.datasets.Omniglot(root="./data/", background=True, download=True)
  imageset = tv.datasets.ImageFolder(root="./data/omniglot-py/images_background/Futurama")

  VAL_CONFIG = {
    "iterations": 100,
    "beta": 1e-4,
    "iterations:batch_size": 32,
  }  # VALIDATION_CONFIG
  VAL_FRAMEWORK = {"n_way": 5, "k_shot": 3, "n_query": 10}
  print(f"Validated Framework: {VAL_FRAMEWORK}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  my_data = torch.load("/content/drive/MyDrive/Colab Notebooks/MAML.bin", map_location=device, weights_only=False)
  my_model = MAML(my_data["MODEL_CONFIG"]).to(device)
  my_model.load_state_dict(my_data["sate"])

  unseen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), my_data["FRAMEWORK"]["n_way"])]
  evisoder = FewShotEpisoder(imageset, unseen_classes, VAL_FRAMEWORK["k_shot"], VAL_FRAMEWORK["n_query"], transform, True)
  counts, n_problems = evaluate(my_model, evisoder=evisoder, config=VAL_CONFIG, device=device, logging=True)
  print(f"unseen classes: {evisoder.classes}\nACC: {(counts / n_problems):.2f}({counts}/{n_problems})")
# if __name__ == "__main__":
```
---
## Technical Highlights
Although MAML is one of the most prominent few-shot learning algorithms, it's mathematically complex even compared to other modern deep learning approaches. Both the learning and evaluation processes consist of two stages.

### Inner Loop
The inner loop is the first stage of MAML's algorithm where task-specific adaptations occur. It involves taking a small number of examples (support set) from a new task and creating parameters for each task. It then performs gradient updates to quickly adapt the model parameters for that specific task.

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
The outer-loop is the second stage of MAML's algorithm where meta-learning occurs. It optimizes the initial model parameters to ensure they can be quickly adapted to new tasks with minimal data. This stage uses performance on the query set to update the model's starting point.

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
The forward process in MAML differs significantly from other deep neural networks. First, it adapts to tasks from the query set. Then, it forwards each parameter per task and calculates probabilities.

```python
def forward(self, x, params=None):
  if not params: params = dict(self.named_parameters())  # uses meta/global params when local params not given
  x = F.conv2d(x, weight=params['conv1.weight'], bias=params['conv1.bias'], padding=self.config["conv:padding"],
               stride=self.config["conv:stride"])
  res = x
  x = F.conv2d(self.act(x) + res, weight=params['conv2.weight'], bias=params['conv2.bias'],
               padding=self.config["conv:padding"], stride=self.config["conv:stride"])
  res = x
  x = F.conv2d(self.act(x) + res, weight=params['conv3.weight'], bias=params['conv3.bias'],
               padding=self.config["conv:padding"], stride=self.config["conv:stride"])
  x = self.pool(x)
  x = self.flatten(x)
  return F.linear(x, weight=params['l1.weight'], bias=params['l1.bias'])
# forward()
```