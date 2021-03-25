import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import os
import PIL
from PIL import Image

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


def get_preprocessed_image(image_path):
    
    # transformations
    preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    #
    # Pass the image for preprocessing and the image preprocessed
    img = Image.open(image_path)
    img_preprocessed = preprocess(img)
    #
    
    batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
    #print('Batch image tensor', batch_img_tensor)
    return batch_img_tensor


model = mask_net()
  # Load state_dict
model.load_state_dict(torch.load('/work/checkpoint3.pth'))


def predict_class(model, batch_tensor):
  # Create the model
  
  model.eval()

  out = model(batch_tensor)
  
  class_idx = out[0].argmax().item()
  classes = {'design_mask': 0, 'design_mask_bad': 1, 'medical_mask': 2, 'medical_mask_bad': 3, 'no_mask': 4}
  
  for k,v in classes.items():
    if v == class_idx:
        return k

PRED_DICT = {'design_mask': 0, 'design_mask_bad': 0, 'medical_mask': 0, 'medical_mask_bad': 0, 'no_mask': 0}

def update_predictions(dir_path):

  file_paths = os.listdir(dir_path)
  base_dir = '/work/Mask_No_Mask/results/Crops/'#'/work/Mask_No_Mask/test_data/'
  for f in file_paths:
    
    preprocessed_img = get_preprocessed_image(base_dir+f) #
    predicted_class = predict_class(model, preprocessed_img)
    PRED_DICT[predicted_class] += 1 
  return PRED_DICT


def plot_bar_chart(title):
  plt.title(title)
  plt.bar(range(len(PRED_DICT)), list(PRED_DICT.values()), align='center')
  plt.xticks(range(len(PRED_DICT)), list(PRED_DICT.keys()), rotation=30)
  return plt