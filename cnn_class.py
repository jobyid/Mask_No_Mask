import torch
import torch.nn as nn
import torch.functional as F

# images are 640 x 640

class mask_net(nn.Module):

    def __init__(self):
        self.conv1 = torch.conv2d()

    def forward(self, x):
        pass
