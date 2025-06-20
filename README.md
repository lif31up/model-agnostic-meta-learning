This implementation is inspired by **"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (2017)**.
* **task**: classifying image with few dataset.
* **dataset**: downloaded from `torch` dataset library.

## Model-Agnostic Meta-Learning for Few-Shot Image Classification
This repository implements a Model-Agnostic Meta-Learning (MAML) algorithm for few-shot image classification tasks using PyTorch.

Few-shot learning focuses on enabling models to generalize to new tasks with only a few labeled examples. MAML achieves this by optimizing for a set of parameters that can quickly adapt to new tasks through gradient-based updates, allowing the model to efficiently learn from limited data.

> Check out the full explanation in [GitBook](https://lif31up.gitbook.io/lif31up/few-shot-learning/model-agnostic-meta-learning-for-fast-adaptation-of-deep-networks)

> You can quick start on [Colab](https://colab.research.google.com/drive/1ZmtP8rMZsSN_yA6tz3IKQU0ECXeAI018#scrollTo=iMjrWpR0FxHn)

> Safetensor on [huggingface](https://huggingface.co/lif31up/model-agnostic-meta-learning)

## Instruction
Organize your dataset into a structure compatible with PyTorch's ImageFolder:
```
dataset/
  ├── class1/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── class2/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  └── ...
 ```

### Training
Run the training script with desired parameters:
```Shell
python run.py train --dataset path/to/your/dataset --save_to /path/to/save/model --n_way 5 --k_shot 2 --n_query 4 --epochs 1 --iters 4
```
* `path`: Path to your dataset.
* `save_to`: path to save the trained model.
* `n_way`: number of classes in each episode.
* `k_shot`: Number of support samples per class.
* `n-_query`: Number of query samples per class.

> change training configuration from `config.py`


### Evaluation
```Shell
python run.py --dataset path/to/your/dataset --model path/to/saved/5w5s.pth --n_way 5
# output example:
# seen classes: [10, 11, 18, 1, 3]
# unseen classes: [19, 1, 4, 2, 13]
# accuracy: 1.0000(10/10)
```
* `model`: Path to your model.
* `dataset`: Path to your dataset.

### Download Omniglot Dataset
```Shell
pyhton download --path ./somewhre/your/dataset/
```
* `path`: Path to your dataset.

---
### More Explanation
**Model-Agnostic Meta-Learning(MAML)** is a powerful approach for few-shot learning, where the objective is to enable rapid adaptation to new tasks with very few labeled examples. The key idea behind MAML is to learn a model initialization that can be quickly fine-tuned on new tasks using only a small number of gradient updates, allowing efficient generalization to unseen data.

* **Task-Specific Adaptation:** Instead of learning fixed representations, MAML optimizes model parameters that can be rapidly adapted to different tasks with gradient-based updates.
* **Inner-Loop Optimization:** For each task, the model is fine-tuned on a small support set using a few gradient steps to minimize task-specific loss.
* **Meta-Update (Outer Loop):** After task-specific updates, gradients are computed based on query set performance, and the initial model parameters are updated to improve adaptability across tasks.
* **Optimization:** The model is trained using second-order gradient updates (or first-order approximations) to optimize for fast adaptation while maintaining generalization ability.

---
### Result

|             | 5 Way ACC (5 shot)   | 5 Way ACC(1 shot)  |
|-------------|----------------------|--------------------|
| **Omniglot** | `100%` **(100/100)** | `96%` **(96/100)** |
