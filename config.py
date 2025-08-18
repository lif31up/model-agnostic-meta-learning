MODEL_CONFIG = {
  "input_channels": 3,
  "hidden_channels": 6,
  "output_channels": 5,
  "conv:kernel_size": 3,
  "conv:padding": 1,
  "conv:stride": 2,
  "l1:in_features": 1944  # insert a x sample in here
}  # MODEL
TRAINING_CONFIG = {
  "iterations": 15,
  "epochs": 25,
  "alpha": 1e-3,
  "beta": 1e-4,
  "iterations:batch_size": 4,
  "epochs:batch_size": 16,
}  # LOOP_CONFIG