import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models


# Define a function to apply CLAHE to an image


def apply_clahe(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    # Convert to grayscale if image has multiple channels
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the image
    clahe_img = clahe.apply(img_array)
    # Convert back to PIL image
    return Image.fromarray(clahe_img)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, ensemble_val_loss, models, paths):
        score = -ensemble_val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(ensemble_val_loss, models, paths)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(ensemble_val_loss, models, paths)
            self.counter = 0

    def save_checkpoint(self, val_loss, models, paths):
        '''Saves both models when the ensemble validation loss decreases.'''
        if self.verbose:
            print(f'Ensemble validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving models ...')
        for model, path in zip(models, paths):
            torch.save(model.state_dict(), path)
        #torch.save(models.state_dict(), paths)    
        self.val_loss_min = val_loss

        
        
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        self.num_ftrs = num_ftrs
        for name, param in model.named_parameters():
            if 'layer2' in name:
                break
            param.requires_grad = False
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        b = x.size(0)
        x = x.view(b, -1)
        return x


class Classifier(nn.Sequential):
    def __init__(self, model_image, **config):
        super(Classifier, self).__init__()
        self.model_image = model_image
        self.input_dim = model_image.num_ftrs
        self.dropout = nn.Dropout(0.5)
        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim] + self.hidden_dims + [11]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

    def forward(self, v):
        v_i = self.model_image(v)
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor)-1):
                v_i = l(v_i)
            else:
                v_i = F.relu(self.dropout(l(v_i)))
        return v_i