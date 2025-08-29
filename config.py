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