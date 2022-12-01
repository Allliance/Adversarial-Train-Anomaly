from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

#######################
#  Define Transform   #
#######################

transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

transform_1_channel = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor()])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)


class Exposure(Dataset):
    def __init__(self, root, extra):
        self.image_files = glob(os.path.join(root, "*.png")) + glob(os.path.join(root, "*.jpeg")) + extra

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = transform(image)

        return image

    def __len__(self):
        return len(self.image_files)

class MergedDataset(Dataset):
    def __init__(self, normal_dataset, exposure_dataset):
        self.normal_dataset = normal_dataset
        self.exposure_dataset = exposure_dataset
    
    def __getitem__(self, index):
        normal_index = index % len(self.normal_dataset)
        exposure_index = index % len(self.exposure_dataset)
        return self.normal_dataset[normal_index][0], 0, self.exposure_dataset[exposure_index], 1
    
    def __len__(self):
        return max(len(self.normal_dataset), len(self.exposure_dataset))

def get_dataloader(dataset='cifar10', normal_class_indx = 0, batch_size=8):
    if dataset == 'cifar10':
        return get_CIFAR10(normal_class_indx, batch_size)
    elif dataset == 'cifar100':
        return get_CIFAR100(normal_class_indx, batch_size)
    elif dataset == 'mnist':
        return get_MNIST(normal_class_indx, batch_size)
    elif dataset == 'fashion-mnist':
        return get_FASHION_MNIST(normal_class_indx, batch_size)
    elif dataset == 'svhn':
        return get_SVHN(normal_class_indx, batch_size)
    else:
        raise Exception("Dataset is not supported yet. ")
     
    
def get_datasets(dataset_class, transform, normal_class_indx):
    try:
        trainset = dataset_class(root=os.path.join('~', 'traindata'), train=True, download=True, transform=transform)
    except:
        trainset = dataset_class(root=os.path.join('~', 'traindata'), split="train", download=True, transform=transform)
    try:
        trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    except:
        trainset.data = trainset.data[np.array(trainset.labels) == normal_class_indx]
    
    if dataset_class == SVHN:
        trainset.labels  = [0 for _ in range(len(trainset.data))]
    else:
        trainset.targets  = [0 for _ in range(len(trainset.data))]
        
    try:
        testset = dataset_class(root=os.path.join('~/', 'testdata'), train=False, download=True, transform=transform)
    except:
        testset = dataset_class(root=os.path.join('~/', 'testdata'), split="test", download=True, transform=transform)
    if dataset_class == SVHN:
        testset.labels  = [int(t!=normal_class_indx) for t in testset.labels]
    else:
        testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    
    return trainset, testset
    
def get_CIFAR100(normal_class_indx, batch_size):
    trainset, testset = get_datasets(CIFAR100, transform, normal_class_indx)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_CIFAR10(normal_class_indx, batch_size):
 
    trainset, testset = get_datasets(CIFAR10, transform, normal_class_indx)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_MNIST(normal_class_indx, batch_size, repeats=1):
    
    trainset, testset = get_datasets(MNIST, transform_1_channel, normal_class_indx)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_FASHION_MNIST(normal_class_indx, batch_size):

    trainset, testset = get_datasets(FashionMNIST, transform_1_channel, normal_class_indx)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_SVHN(normal_class_indx, batch_size):

    trainset, testset = get_datasets(SVHN, transform, normal_class_indx)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
