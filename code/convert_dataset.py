import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset 
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb

# pull images from places365/val_256 in directory
def getImages():
    folder_path = Path("../places365/val_256")
    return sorted(folder_path.glob("*.jpg"))

# class to convert color images to lab
class ColorizeDataset(Dataset):
    # initialize images
    def __init__(self, image_paths, image_size=128):
        self.image_paths = image_paths
        self.image_size = image_size

    # length of the image
    def __len__(self):
        return len(self.image_paths)

    # get the image and convert RGB to Lab
    def __getitem__(self, i): 
        image = Image.open(self.image_paths[i]).convert("RGB")
        image = image.resize((self.image_size, self.image_size))

        # convert from RGB to Lab
        lab = rgb2lab(np.array(image)).astype("float32")

        # normalize to about [-1,1]
        L = lab[:, :, 0] / 50.0 - 1.0
        ab = lab[:, :, 1:] / 110.0

        # to tensors with channels first: L is (1,H,W) and ab is (2,H,W)
        L = torch.from_numpy(L).unsqueeze(0)
        ab = torch.from_numpy(ab).permute(2, 0, 1)

        return L, ab
    
# convert normalized L and ab tensors back to RGB images for display
def lab_tensors_to_rgb(L, ab):
    # L: (B,1,H,W) in [-1,1]    ab: (B,2,H,W) in [-1,1]
    L = (L.detach().cpu().numpy() + 1.0) * 50.0
    ab = ab.detach().cpu().numpy() * 110.0
    lab = np.concatenate([L, ab], axis=1).transpose(0, 2, 3, 1).astype("float64")
    return np.stack([np.clip(lab2rgb(img), 0, 1) for img in lab])