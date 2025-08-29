import torchvision as tv

transform = tv.transforms.Compose([
  tv.transforms.Resize((28, 28)),
  tv.transforms.Grayscale(num_output_channels=1),
  tv.transforms.ToTensor(),
  tv.transforms.Normalize(mean=[0.5], std=[0.5]),
]) # transform