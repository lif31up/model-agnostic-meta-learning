# MAML and Its Variants for FSL Image Classification

This implementation is inspired by:
[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.05175) by Jake Snell, Kevin Swersky, Richard S. Zemel.
[BOIL: Towards Representation Change for Few-shot Learning](https://arxiv.org/abs/2008.08882) by Jaehoon Oh, Hyungjun Yoo, ChangHwan Kim, Se-Young Yun.
[Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML](https://arxiv.org/abs/1909.09157) by Aniruddh Raghu, Maithra Raghu, Samy Bengio, Oriol Vinyals.

FSL (Few-Shot Learning) enables models to generalize to new tasks with only a few labeled examples. MAML achieves this by optimizing parameters that can quickly adapt to new tasks through gradient-based updates, allowing the model to learn efficiently from limited data.

* **Task**: classifying image with few dataset.
* **Dataset**: `omniglot futurama`

### Requirements
To run the code on your own machine, run `pip install -r requirements.txt`.

### Configuration
`confing.py` contains the configuration settings for the model, including the framework, dimensions, learning rates (alpha, beta), and other hyperparameters like kernel size.

```python
class Config:
  def __init__(self):
    self.input_channels, self.hidden_channels, self.output_channels = 1, 32, 5
    self.n_convs = 4
    self.kernel_size, self.padding, self.stride, self.bias = 3, 1, 1, True
    self.iterations, self.alpha = 100, 1e-3
    self.eps = 1e-5
    self.epochs, self.beta = 30, 1e-4
    self.batch_size = 8
    self.n_way, self.k_shot, self.n_query = 5, 5, 5
    self.save_to = "./models"
    self.transform = transform
    self.imageset = get_imageset()
    self.dummy = torch.zeros(1, self.input_channels, 28, 28)
```
### Training
`train.py` is a script to train the model on the omniglot dataset. It includes the training loop and saving the model checkpoints.
```python
if __name__ == "__main__":
  from config import Config
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  maml_config = Config()
  imageset = maml_config.imageset
  seen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), maml_config.n_way)]
  episoder = FewShotEpisoder(imageset, seen_classes, maml_config.k_shot, maml_config.n_query, maml_config.transform)
  model = ResNetMAML(maml_config)  # choose your arch from here!!
  train(path=maml_config.save_to, model=model, config=maml_config, episoder=episoder, device=device)
```
### Evaluation
`eval.py` is used to evaluate the trained model on the omniglot dataset. It loads the model and tokenizer, processes the dataset, and computes the accuracy of the model.
```python
if __name__ == "__main__":
  from config import Config
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  my_data = torch.load("put your model path!!", map_location=device, weights_only=False)
  config, convig = my_data['config'], Config()
  my_model = ResNetMAML(config).to(device)
  my_model.load_state_dict(my_data["sate"])
  unseen_classes = [_ for _ in random.sample(list(convig.imageset.class_to_idx.values()), my_data["FRAMEWORK"]["n_way"])]
  evisoder = FewShotEpisoder(convig.imageset, unseen_classes, convig.k_shot, convig.n_query, config.transform, True)
  counts, n_problems = evaluate(my_model, evisoder=evisoder, config=config, device=device, logging=True)
  print(f"unseen classes: {evisoder.classes}\nACC: {(counts / n_problems):.2f}({counts}/{n_problems})")
```