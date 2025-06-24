def evaluates(MODEL: str, DATASET: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"the device type of {device.type}.")

  # load model
  data = torch.load(MODEL)
  model_config = (data["MODEL_CONFIG"]["in_channels"], data["MODEL_CONFIG"]["hidden_channels"], data["MODEL_CONFIG"]["output_channels"], (15, data["HYPER_PARAMETERS"]["alpha"]))
  model = MAML(*model_config).to(device)
  model.load_state_dict(data["state"])

  # overall configuration
  n_way, k_shot, n_query = data["FRAMEWORK"].values()
  transform = data["TRANSFORM"]

  # evaluate
  accuracy, n_eval = 0., 10
  count, n_problem, progress_bar = 0, 0, tqdm(range(n_eval), desc="evaluation")
  for _ in progress_bar:
    # create FewShotEpisoder which creates tuple of (support set, query set)
    imageset = tv.datasets.ImageFolder(root=DATASET)
    unseen_classes = random.sample(list(imageset.class_to_idx.values()), n_way)
    evisoder = FewShotEpisoder(imageset, unseen_classes, k_shot, n_query, transform)

    # fast adaption(inner loop)
    (tasks, query_set), adaptions = evisoder.get_episode(), list()
    for task in tasks: adaptions.append(model.inner_update(task, device))

    # evaluate
    for feature, label in DataLoader(query_set, shuffle=True):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      task_i = torch.argmax(label).item()
      pred = model.forward(feature, adaptions[task_i])
      if torch.argmax(pred) == torch.argmax(label): count += 1
      n_problem += 1
    # for
    accuracy += count / n_problem
    progress_bar.set_postfix(accuracy=count / n_problem)
  print(f"seen classes: {data['seen_classes']}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
# main