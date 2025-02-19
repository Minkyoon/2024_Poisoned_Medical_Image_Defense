import os
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from src.utils import apply_clahe, EarlyStopping
from models.ETFModel import ETFModel
import copy

# Set random seed and device
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2022)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory for results
result_dir = './results/240819_2305_resnet50_ETF_dml_react_earaly_stop_true_deepcopy'
os.makedirs(result_dir, exist_ok=True)

# Helper functions for plotting
def plot_loss_curve(loss_history_train, loss_history_val, save_path):
    plt.figure(figsize=(10, 7))
    plt.plot(loss_history_train, label='Train')
    plt.plot(loss_history_val, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    plt.close()

# Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, int(label)

# Model definitions
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        model = models.resnet50(pretrained=True)
        self.num_ftrs = model.fc.in_features
        for name, param in model.named_parameters():
            if 'layer2' in name:
                break
            param.requires_grad = False
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class Classifier(nn.Sequential):
    def __init__(self, model_image, **config):
        super(Classifier, self).__init__()
        self.model_image = model_image
        self.input_dim = model_image.num_ftrs
        self.hidden_dims = config['cls_hidden_dims']
        dims = [self.input_dim] + self.hidden_dims + [11]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, v):
        v_i = self.model_image(v)
        for i, layer in enumerate(self.predictor):
            v_i = layer(v_i) if i == len(self.predictor) - 1 else F.relu(layer(v_i))
        return v_i

# DML Trainer class
class DMLTrainer:
    def __init__(self, models, optimizers, train_loader, valid_loader, num_epochs, device):
        self.models = models
        self.optimizers = optimizers
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.device = device
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.schedulers = [optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, verbose=True) for opt in optimizers]
        self.early_stopping = EarlyStopping(patience=20, verbose=True)
        self.loss_history_train = [[] for _ in models]
        self.loss_history_val = [[] for _ in models]

    def train_one_epoch(self, epoch):
        for model in self.models:
            model.train()

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            for i, model in enumerate(self.models):
                features = model.feature_extractor(data)
                logits = model.fc(features)
                ce_loss = self.loss_ce(logits, target)
                kl_loss = sum(self.loss_kl(F.log_softmax(logits, dim=1), F.softmax(model.fc(model.feature_extractor(data)), dim=1)) for model in self.models if model != self.models[i])
                loss = ce_loss + kl_loss / (len(self.models) - 1)
                optimizer = self.optimizers[i]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.loss_history_train[i].append(loss.item())

    def validate(self, epoch):
        val_losses = [0 for _ in self.models]
        for model in self.models:
            model.eval()

        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                for i, model in enumerate(self.models):
                    logits = model.fc(model.feature_extractor(data))
                    ce_loss = self.loss_ce(logits, target)
                    kl_loss = sum(self.loss_kl(F.log_softmax(logits, dim=1), F.softmax(model.fc(model.feature_extractor(data)), dim=1)) for model in self.models if model != self.models[i])
                    loss = ce_loss + kl_loss / (len(self.models) - 1)
                    val_losses[i] += loss.item()
                    self.loss_history_val[i].append(loss.item())

        return [loss / len(self.valid_loader) for loss in val_losses]

    def train(self):
        paths = [os.path.join(result_dir, f'best_model_{i}.pth') for i in range(len(self.models))]
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            val_losses = self.validate(epoch)
            ensemble_val_loss = np.mean(val_losses)
            self.early_stopping(ensemble_val_loss, self.models, paths)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
            for scheduler, val_loss in zip(self.schedulers, val_losses):
                scheduler.step(val_loss)

    def test(self, test_loader):
        y_true, y_preds, y_scores = [], [], []
        for model in self.models:
            model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                y_true.extend(target.cpu().numpy())
                for model in self.models:
                    logits = model(data)
                    scores = F.softmax(logits, dim=1)
                    preds = scores.argmax(dim=1)
                    y_scores.append(scores.cpu().numpy())
                    y_preds.append(preds.cpu().numpy())

        ensemble_preds = np.argmax(np.mean(y_scores, axis=0), axis=1)
        print(classification_report(y_true, ensemble_preds))

# Dataset and data loaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_params = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 1,
    'drop_last': False
}

csv_dir = 'csv_dir'

# Initialize and train models
feature_extractor = ResNet()
model1 = ETFModel(num_classes=11, feature_extractor=feature_extractor).to(device)
model2 = copy.deepcopy(model1)
models = [model1, model2]
optimizers = [optim.Adam(model.parameters(), lr=0.0001) for model in models]

train_dataset = CustomImageDataset(os.path.join(csv_dir, 'train_fold0.csv'), transform=transform)
valid_dataset = CustomImageDataset(os.path.join(csv_dir, 'val_fold0.csv'), transform=transform)
train_loader = DataLoader(train_dataset, **train_params)
valid_loader = DataLoader(valid_dataset, **train_params)

trainer = DMLTrainer(models, optimizers, train_loader, valid_loader, num_epochs=100, device=device)
trainer.train()
