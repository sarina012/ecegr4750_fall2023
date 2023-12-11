from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch
import math
import os
from pathlib import Path
from torchvision import transforms

# Create CustomDataloader class
class CustomDataloader:
    """
    Wraps a dataset and enables fetching of one batch at a time
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, randomize: bool = False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)
    
    def __len__(self):
        if self.randomize:
            return self.num_batches_per_epoch
        else:
            return 1
        
    def get_length(self):
        return self.x.shape[0]

    def randomize_dataset(self):
        """
        This function randomizes the dataset, while maintaining the relationship between 
        x and y tensors
        """
        indices = torch.randperm(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]

    def generate_iter(self):
        """
        This function converts the dataset into a sequence of batches, and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time
        """
        if self.randomize:
            self.randomize_dataset()

        # split dataset into a sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append({
                'x_batch': self.x[b_idx * self.batch_size: (b_idx + 1) * self.batch_size],
                'y_batch': self.y[b_idx * self.batch_size: (b_idx + 1) * self.batch_size],
                'batch_idx': b_idx,
            })
        self.iter = iter(batches)
    
    def fetch_batch(self):
        """
        This function calls next on the batch iterator and also detects when the final batch
        has been run, so that the iterator can be re-generated for the next epoch
        """
        # if the iter hasn't been generated yet
        if self.iter is None:
            self.generate_iter()

        # fetch the next batch
        batch = next(self.iter)

        # detect if this is the final batch to avoid StopIteration error
        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iter
            self.generate_iter()
        return batch
    
class NeuralNetworkData(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['Filepath']
        age = self.dataframe.iloc[idx]['Age']

        # Load the image
        img = Image.open(img_path).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)

        return {'image': img, 'age': age}


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)

        # Convert the 'age' column to numeric, forcing non-convertible values to NaN, then drop those rows
        self.data_frame['age'] = pd.to_numeric(self.data_frame['age'], errors='coerce')
        self.data_frame.dropna(subset=['age'], inplace=True)

        # Convert 'yes'/'no' columns to binary (0/1)
        binary_columns = ['has_tiktok', 'remembers_disco', 'uses_skincare']
        for col in binary_columns:
            self.data_frame[col] = self.data_frame[col].map({'yes': 1, 'no': 0})

        # One-hot encode 'gender', 'race', and 'age_range' columns
        self.data_frame = pd.get_dummies(self.data_frame, columns=['gender', 'race', 'age_range'])

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx]['filename']))
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Get all features excluding 'filename' and 'age' (which is the target variable)
        features = self.data_frame.drop(columns=['filename', 'age']).iloc[idx].values
        # Convert non-numeric values to floats
        features = [float(val) if isinstance(val, (int, float)) else 0.0 for val in features]
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Get the 'age' column by name to ensure correct data is retrieved and it's already numeric
        label = self.data_frame.iloc[idx]['age']
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return image, features_tensor, label_tensor
    
