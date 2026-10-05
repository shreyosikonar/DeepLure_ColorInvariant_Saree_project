import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

class SareeEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=256, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            m = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        in_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone=m
        self.head=nn.Sequential(nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(512,embedding_dim))
    def forward(self,x):
        z=self.head(self.backbone(x))
        return torch.nn.functional.normalize(z,p=2,dim=1)
