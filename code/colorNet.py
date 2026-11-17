import torch
import torch.nn as nn

# 3x3 conv -> BatchNorm -> Relu. twice
def convBlock(input, output):
    return nn.Sequential(
        nn.Conv2d(input, output, 3, padding=1),
        nn.BatchNorm2d(output),
        nn.ReLU(inplace=True),

        nn.Conv2d(output, output, 3, padding=1),
        nn.BatchNorm2d(output),
        nn.ReLU(inplace=True),
    )

class ColorNet(nn.Module):
    def __init__(self, mode):
        super(ColorNet, self).__init__()

        # encoder used by all 3 modes
        self.enc1 = convBlock(1, 32)                 # H 
        self.enc2 = convBlock(32, 32 * 2)           # H/2
        self.enc3 = convBlock(32 * 2, 32 * 4)        # H/4
        self.bottle_neck = convBlock(32 *4, 32 * 8)  # H/8
        self.pool = nn.MaxPool2d(2)

        # decoder (upsampling layers)
        self.up1 = nn.ConvTranspose2d(32 * 2, 32, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(32 * 4, 32 * 2, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(32 * 8, 32 * 4, 2, stride=2)

        # skip connections and double the channels going into the decoder
        if mode >= 2:
            k = 2
        else:
            k = 1

        self.dec1 = convBlock(32 * k, 32)
        self.dec2 = convBlock(32 * 2 * k, 32 * 2)
        self.dec3 = convBlock(32 * 4 * k, 32 * 4)

        self.out = nn.Conv2d(32, 2, 1)
        self.act = nn.Tanh()

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-3")
            exit(0)

    # baseline model: encoder + decoder, no skips
    def model_1(self, X):
        # encode
        X = self.enc1(X)
        X = self.enc2(self.pool(X))
        X = self.enc3(self.pool(X))
        X = self.bottle_neck(self.pool(X))

        # decode
        X = self.dec3(self.up3(X))
        X = self.dec2(self.up2(X))
        X = self.dec1(self.up1(X))
        X = self.act(self.out(X))

        return X

    # U-net: encoder + decoder, concatenates encoder features into decoder
    def model_2(self, X):
        # encode
        e1 = self.enc1(X)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottle = self.bottle_neck(self.pool(e3))

        # decode
        X = self.dec3(torch.cat([self.up3(bottle), e3], 1))
        X = self.dec2(torch.cat([self.up2(X), e2], 1))
        X = self.dec1(torch.cat([self.up1(X), e1], 1))
        X = self.act(self.out(X))

        return X

    def model_3(self, X):
            return NotImplementedError()