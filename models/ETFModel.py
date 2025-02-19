import torch.nn as nn
from models.MLPFFNNeck import MLPFFNNeck
from .ETFClassifier import ETFClassifier




class ETFModel(nn.Module):
    def __init__(self, num_classes, feature_extractor, in_channels=2048, out_channels=2048):
        super(ETFModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = MLPFFNNeck(in_channels, out_channels)
        self.classifier = ETFClassifier(num_classes, out_channels)

    def forward(self, x):
        features = self.feature_extractor(x)
        projections = self.fc(features)
        logits = self.classifier(projections)
        return logits