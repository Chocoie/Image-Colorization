import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
import numpy as np 
from colorNet import ColorNet
from convert_dataset import ColorizeDataset, getImages, lab_tensors_to_rgb
import matplotlib.pyplot as plt
import random
import time

class ChromaLoss(nn.Module):
    """L1 on ab, plus a penalty when predicted chroma differs from the true chroma."""
    def __init__(self, weight=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.weight = weight

    def forward(self, pred, target):
        pred_c = torch.sqrt((pred ** 2).sum(1) + 1e-8)    # chroma = sqrt(a^2 + b^2)
        true_c = torch.sqrt((target ** 2).sum(1) + 1e-8)
        return self.l1(pred, target) + self.weight * (pred_c - true_c).abs().mean()

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    start = time.time()
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    
    # Iterate over entire training samples (1 epoch)
    for L, ab in train_loader:
        # Push data/label to correct device
        L = L.to(device)
        ab = ab.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()

        # prediction for ab
        pred_ab = model(L)

        # loss between predicted and true ab
        loss = criterion(pred_ab, ab) 
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()

        if len(losses) % 20 == 0:
            print(f"  batch {len(losses)}/{len(train_loader)} | {time.time() - start:.0f}s")
        
    train_loss = float(np.mean(losses))
    print(f"Epoch {epoch} | train loss: {train_loss:.4f}")
    return train_loss

def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    ab_mean =[] # absolute mean error
    gray_mean = [] # absolute mean error of the "predict gray" baseline
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for L, ab in test_loader:
            L, ab = L.to(device), ab.to(device)
            
            # # prediction for ab
            pred_ab = model(L)

            # calculate loss
            loss = criterion(pred_ab, ab)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            ab_mean.append((pred_ab - ab).abs().mean().item() * 110.0)
            gray_mean.append(ab.abs().mean().item() * 110.0)

    test_loss = float(np.mean(losses))
    test_mean = float(np.mean(ab_mean))
    gray_error = float(np.mean(gray_mean))
    print(f" Test loss: {test_loss:.4f} | absolute mean error: {test_mean:.2f} | gray error: {gray_error:.2f}")
    return test_loss, test_mean

def save_samples(model, device, dataset, path, n=6):
    model.eval()
    i = list(range(min(n, len(dataset))))
    L = torch.stack([dataset[j][0] for j in i]).to(device)
    ab = torch.stack([dataset[j][1] for j in i])

    with torch.no_grad():
        pred = model(L).cpu()

    L = L.cpu()
    grayImg = lab_tensors_to_rgb(L, torch.zeros_like(ab))
    predImg = lab_tensors_to_rgb(L, pred)
    realImg = lab_tensors_to_rgb(L, ab)

    fix, axes = plt.subplots(3, len(i), figsize=(2.5 * len(i), 7.5))
    axes = np.atleast_2d(axes).reshape(3, -1)

    for k in range(len(i)):
        for r, (img, title) in enumerate([(grayImg, "Input (L)"), (predImg, "Predicted"), (realImg, "Ground Truth")]):
            axes[r, k].imshow(img[k])
            axes[r, k].axis("off")

            if k == 0:
                axes[r, k].set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    
def run_main(FLAGS):
    st = time.perf_counter()
    # random seeds
    random.seed(0)
    torch.manual_seed(0)

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)

    os.makedirs("outputs", exist_ok=True)

    # get data: image paths, split into training and test samples
    paths = getImages()
    random.Random(0).shuffle(paths)
    if FLAGS.max_images is not None:
        paths = paths[:FLAGS.max_images]
    n_test = max(1, int(len(paths) * 0.1))
    test_paths = paths[:n_test]
    train_paths = paths[n_test:]
    print(f"{len(train_paths)} train images, {len(test_paths)} test images")

    train_set = ColorizeDataset(train_paths, 128)
    test_set = ColorizeDataset(test_paths, 128)
    train_loader = DataLoader(train_set, batch_size=FLAGS.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=FLAGS.batch_size,
                                  shuffle=False, num_workers=0)
    
    # Initialize the model and send to device 
    model = ColorNet(FLAGS.mode).to(device)

    # loss function depends on the mode
    if FLAGS.mode == 3:
        criterion = ChromaLoss(FLAGS.chroma_weight)
    else:
        criterion = nn.L1Loss()
    
    # optimizer function
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    best_error = float("inf")
    tag = "_test" if FLAGS.max_images is not None else ""
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss, test_error = test(model, device, test_loader, criterion)
        
        if test_error < best_error:
            best_error = test_error

            torch.save(model.state_dict(), os.path.join("outputs", f"colornet_mode{FLAGS.mode}{tag}.pt"))

        if epoch % 5 == 0 or epoch == FLAGS.num_epochs:
            save_samples(model, device, test_set,
                         os.path.join("outputs", f"mode{FLAGS.mode}{tag}_samples_epoch{epoch}.png"))
    
    
    print("accuracy is {:2.2f}".format(best_error))
    
    print("Training and evaluation finished")

    et = time.perf_counter()
    # Show time just to see
    amountTime = et - st
    print(f"Execution time: {amountTime:.6f} seconds")
    
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Image colorization.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-3.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=20,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=32,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--max_images',
                            type=int, default=None,
                            help='Use only the first N images.')
    parser.add_argument('--chroma_weight',                
                        type=float, default=0.5,
                        help='Weight of the chroma penalty (mode 3 only).')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    