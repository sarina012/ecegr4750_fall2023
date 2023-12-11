import os
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

# CNN Model for Regression
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)  
        self.fc1 = nn.Linear(32 * 50 * 50, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 50 * 50)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x 

class MultiModalCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MultiModalCNN, self).__init__()
        # Convolutional layers for images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Additional Conv layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Dropout layer

        # Image feature to vector
        self.fc1_img = nn.Linear(128 * 32 * 32, 512)  # Adjust for the new layer

        # Tabular data feature vector
        self.fc1_tab = nn.Linear(num_features, 256)

        # Combined feature vector
        self.fc2_combined = nn.Linear(512 + 256, num_classes)

    def forward(self, x_img, x_tab):
        # Image features
        x_img = self.pool(F.relu(self.conv1(x_img)))
        x_img = self.pool(F.relu(self.conv2(x_img)))
        x_img = self.pool(F.relu(self.conv3(x_img)))  # Pass through additional layer
        x_img = x_img.view(-1, 128 * 32 * 32)  # Adjust for the new layer
        x_img = F.relu(self.fc1_img(x_img))
        x_img = self.dropout(x_img)  # Apply dropout

        # Tabular features
        x_tab = F.relu(self.fc1_tab(x_tab))

        # Combine image and tabular features
        x = torch.cat((x_img, x_tab), dim=1)
        x = F.relu(self.fc2_combined(x))
        return x
