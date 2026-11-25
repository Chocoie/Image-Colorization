# import the data into the current working directory
from torchvision.datasets import Places365
Places365(root="./places365", split="val", small=True, download=True)