import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torchvision import models

# images are 640 x 640
#num_ftrs = 0
class mask_net(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet18(pretrained=True)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, 6)
        #self.conv1 = torch.conv2d()

    def forward(self, x):
        out = self.model_ft.fc(x)
        return out


model = mask_net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
