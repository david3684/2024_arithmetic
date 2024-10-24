import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 test_size=0.2,
                 random_state=42, num_test_samples=None):
        # Data loading code
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Download and load the SUN397 dataset
        print("Downloading and loading the SUN397 dataset...")
        full_dataset = datasets.SUN397(root=self.location, transform=preprocess, download=True)

        # Split the dataset into training and testing sets
        train_idx, test_idx = train_test_split(list(range(len(full_dataset))), test_size=test_size, random_state=random_state)
        if num_test_samples is not None:
            test_idx = test_idx[:num_test_samples]
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        self.test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

        # Ensure indices are within bounds
        assert max(train_idx) < len(full_dataset), "Train indices are out of range!"
        assert max(test_idx) < len(full_dataset), "Test indices are out of range!"

        # Create train loader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False 
        )

        # Map class indices to class names
        idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
        self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]

        # Debugging info: print class names and dataset size
        print(f"Number of classes: {len(self.classnames)}")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
