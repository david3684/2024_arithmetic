import os
import torch
import torchvision.datasets as datasets
from torchvision.datasets import DTD as TorchvisionDTD
from torch.utils.data.sampler import SubsetRandomSampler

class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 num_test_samples=None):  # num_test_samples 추가
        # Data loading code
        self.train_dataset = TorchvisionDTD(
            root=location, split='train', transform=preprocess, download=True)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = TorchvisionDTD(
            root=location, split='val', transform=preprocess, download=True)

        if num_test_samples is not None:
            # num_test_samples가 실제 test 데이터 개수보다 많을 경우 가능한 최대를 사용
            num_test_samples = min(num_test_samples, len(self.test_dataset))
            test_sampler = SubsetRandomSampler(range(num_test_samples))
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=test_sampler
            )
        else:
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )

        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        
        print(f"Number of classes: {len(self.classnames)}")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")