
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torchvision import models








class mask_net(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet18(pretrained=True)
        #for param in self.model_ft.features.parameters():
         #   param.requires_grad = False
        self.num_ftrs = self.model_ft.fc.in_features
        layers = list(self.model_ft.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(self.num_ftrs, 5)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        out = self.classifier(representations)

        return out



model = mask_net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)