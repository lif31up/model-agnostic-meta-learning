FRAMEWORK = { "n_way": 5, "k_shot": 5, "n_query": 2 }
HYPER_PARAMETERS = { "alpha": 0.01, "beta": 0.0001 }
TRAINING_CONFIG = { "iters": 5, "epochs": 10, "batch_size": 8 }
MODEL_CONFIG = { "in_channels": 3, "hidden_channels": 6, "output_channels": 5 } # change number of params this location: MAML.py(12 line)