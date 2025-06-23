This implementation is inspired by [**"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"**](https://arxiv.org/abs/1703.05175) (2017) by Jake Snell, Kevin Swersky, Richard S. Zemel.
* **Note & Reference:** [GitBook](https://lif31up.gitbook.io/lif31up/few-shot-learning/model-agnostic-meta-learning-for-fast-adaptation-of-deep-networks)
* **Quickstart on Colab:** [Colab]()

|            | 5 Way ACC (5 shot) | 5 Way ACC(1 shot) |
|------------|--------------------|------------------|
|**Omniglot**|`100%` **(100/100)**|`96%` **(96/100)**|

## Model-Agnostic Meta-Learning for Few-Shot Image Classification
This repository implements a Model-Agnostic Meta-Learning (MAML) algorithm for few-shot image classification tasks using PyTorch. MAML is designed to address the challenge of adapting to new tasks with limited examples by learning an initialization that enables fast adaptation with minimal gradient steps.

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
  from datasets import load_dataset

  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  train(dataset, config=CONFIG, SAVE_TO="BERT")  # Replace with the actual model path
# __name__
```
### Evaluation
eval.py is used to evaluate the trained model on the omniglot dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.
```python
if __name__ == "__main__":
  from datasets import load_dataset

  dataset = load_dataset('imdb')['train'].shuffle(seed=42).select(range(100))
  evaluate("BERT.pth", dataset)  # Replace with the actual model path
# __name__
## output example: accuracy: 0.91
```