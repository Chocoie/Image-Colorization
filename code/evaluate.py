import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from colorNet import ColorNet
from convert_dataset import ColorizeDataset, getImages

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int, default=1)
parser.add_argument("--weights", type=str, default=None,
                    help="Path to a .pt file. Default: outputs/colornet_mode<mode>.pt")
args = parser.parse_args()

weights = args.weights or os.path.join("outputs", f"colornet_mode{args.mode}.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same split as train.py: same shuffle seed, first 10% is the test set
paths = getImages()
random.Random(0).shuffle(paths)
n_test = max(1, int(len(paths) * 0.1))
test_paths = paths[:n_test]
print(f"{len(test_paths)} test images")

test_loader = DataLoader(ColorizeDataset(test_paths, 128), batch_size=32, shuffle=False)

model = ColorNet(args.mode).to(device)
model.load_state_dict(torch.load(weights, map_location=device))
model.eval()

model_err, gray_err, pred_c, true_c = [], [], [], []
with torch.no_grad():
    for L, ab in test_loader:
        L, ab = L.to(device), ab.to(device)
        pred = model(L)
        model_err.append((pred - ab).abs().mean().item() * 110.0)
        gray_err.append(ab.abs().mean().item() * 110.0)
        # chroma = distance from gray, sqrt(a^2 + b^2)
        pred_c.append(torch.sqrt((pred ** 2).sum(1) + 1e-8).mean().item() * 110.0)
        true_c.append(torch.sqrt((ab ** 2).sum(1) + 1e-8).mean().item() * 110.0)

print(f"Weights: {weights}")
print(f"Mode {args.mode} ab error: {np.mean(model_err):.2f}")
print(f"Gray baseline error: {np.mean(gray_err):.2f}")
print(f"Mean chroma, predicted: {np.mean(pred_c):.2f} | ground truth: {np.mean(true_c):.2f}")