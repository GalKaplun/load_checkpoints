"""
A utility file to read data (e.g., ImageNet) and load it to a pytorch Dataset.
"""
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np

def get_imagenet(imagenet_path, train, no_transform=False):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if train:
        transform = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]
        )
    if no_transform:
        transform = None
    
    data_path = os.path.join(data_path, 'train' if train else 'val')
    # data_path = os.path.join(data_path, 'train_fixed_size' if train else 'val_fixed_size')
    return torchvision.datasets.ImageFolder(root=data_path, transform=transform)