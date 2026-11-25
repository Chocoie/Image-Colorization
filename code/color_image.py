import argparse
import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib
matplotlib.use("Agg") # draw figures to files without opening a popup window
import matplotlib.pyplot as plt
from colorNet import ColorNet

# size the model was trained on
SIZE = 128
MAX_SIDE = 1024 # shrink large photos os Lab conversion is fast

# color the image and return gray, colorized, and original versions
def colorize(model, device, path):
    # convert images to RGB
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    width, height = img.size
    full = np.array(img)

    # full resolution lightness, kept as is for the final image
    LFull = rgb2lab(full)[:, :, 0]

    # mode input
    # L channel is scaled to [-1,1] like in training
    small = np.array(img.resize((SIZE, SIZE)))
    LSmall = rgb2lab(small)[:, :, 0].astype("float32") / 50.0 - 1.0
    x = torch.from_numpy(LSmall)[None, None].to(device)

    # predict the color channels
    with torch.no_grad():
        ab = model(x)

    # unsample the predicted colors to the original size and undo /110 scaling
    ab = F.interpolate(ab, size=(height, width), mode="bilinear", align_corners=False)
    ab = ab[0].cpu().numpy().transpose(1, 2, 0) * 110.0

    # combine the roginal full resolution L with the predicted ab
    colorized = np.clip(lab2rgb(np.concatenate([LFull[:, :, None], ab], axis=2)), 0, 1)

    # grayscale version for display
    gray = np.clip(lab2rgb(np.concatenate([LFull[:, :, None], np.zeros_like(ab)], axis=2)), 0, 1)

    return gray, colorized, full

def main():
    # arguments to run color_image
    parser = argparse.ArgumentParser("Colorize you own images.")
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--weights", type=str, default=None, 
                        help="Path to a .pt file. Default: outputs/colornet_mode<mode>.pt")
    parser.add_argument("--input", type=str, required=True,
                        help="An image file or a folder of images.")
    parser.add_argument("--out_dir", type=str, default="colorized")
    args = parser.parse_args()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")

    weights = args.weights or os.path.join("outputs", f"colornet_mode{args.mode}.pt")

    model = ColorNet(args.mode).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # get image paths either from folder or single image
    if os.path.isdir(args.input):
        paths = []
        for ext in ("*.jpg", "*.jpeg", ".png"):
            paths += glob.glob(os.path.join(args.input, ext))
        paths = sorted(paths)
    else:
        paths = [args.input]

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"mode{args.mode}"

    for p in paths:
        gray, colorized, original = colorize(model, device, p)
        name = os.path.splitext(os.path.basename(p))[0]

        # save colorized image on its own
        Image.fromarray((colorized * 255).astype("uint8")).save(
            os.path.join(args.out_dir, f"{name}_{tag}_color.png"))

        # plot to show input, colorized, and original
        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        for ax, im, title in zip(axes, [gray, colorized, original], 
                                 ["Input (L Only)", f"Colorized ({tag})", "Original"]):
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{name}_{tag}_compare.png"), dpi=120)
        plt.close(fig)
        print("Save", name)

if __name__ == "__main__":
    main()