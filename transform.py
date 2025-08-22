import torchvision as tv

transform = tv.transforms.Compose([
  tv.transforms.Resize((222, 222)),
  tv.transforms.ToTensor(),
  tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])  # transform