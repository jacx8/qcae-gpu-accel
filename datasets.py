import torch
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from constant import *

#def train_dataset(n_samples = 20, batch_size = 1):
#    X_train = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.Resize((28,28)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
#    
#    # Convert X_train.targets to a tensor if it's not already
#    X_train.targets = torch.tensor(X_train.targets, dtype=torch.long)  # Ensure it's of type long (int64)
#    
#    # Print targets to confirm it's a tensor
#    print("Targets:", X_train.targets)
#    
#    # Get indices for class 0 and class 1
#    idx_0 = torch.where(X_train.targets == 2)[0][:n_samples]
#    #idx_1 = torch.where(X_train.targets == 1)[0][:n_samples]
#    
#    # Check the indices for class 0 and class 1
#    print(f"Indices for class 0: {idx_0}")
#    #print(f"Indices for class 1: {idx_1}")
#    
#    # Combine indices for class 0 and class 1
#    #idx = torch.cat([idx_0, idx_1])
#    idx = torch.cat([idx_0])
#
#    # Convert the indices to a list of integers (if necessary)
#    idx = idx.tolist()  # Convert to a list
#    
#    # Check the final idx
#    print("Combined IDX IS: ", idx)
#    
#    # Index X_train.data and X_train.targets with the list of integer indices
#    X_train.data = X_train.data[idx]
#    X_train.targets = X_train.targets[torch.tensor(idx, dtype=torch.long)]
#
#    # Create DataLoader
#    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False, pin_memory=True)
#    return train_loader
#
#
#
#def test_dataset(n_samples = 20, batch_size = 1):
#    X_test = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.Resize((28,28)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
#    
#    # Convert X_test.targets to a tensor if it's not already
#    X_test.targets = torch.tensor(X_test.targets, dtype=torch.long)  # Ensure it's of type long (int64)
#    
#    # Get indices for class 0 and class 1
#    idx_0 = torch.where(X_test.targets == 2)[0][:n_samples]
#    #idx_1 = torch.where(X_test.targets == 1)[0][:n_samples]
#    
#    # Combine indices for class 0 and class 1 (convert idx to tensor if needed)
#    #idx = torch.cat([idx_0, idx_1])
#    idx = torch.cat([idx_0])
#    
#    # Index X_test.data and X_test.targets with the tensor of integer indices
#    X_test.data = X_test.data[idx]
#    X_test.targets = X_test.targets[idx]
#    
#    # Create DataLoader
#    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
#    return test_loader

def train_dataset(n_samples = 200, batch_size = 1):
    X_train = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),transforms.ToTensor()]))

    # # Leaving only labels 0 and 1 
    idx = np.where(X_train.targets == 1)[0][:n_samples]
        #np.where(X_train.targets == 1)[0][:n_samples])
    # idx = np.stack([np.where(X_train.targets == i)[0][:n_samples] for i in range(10)], axis=1)
    # idx = idx.reshape(-1)

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader


def test_dataset(n_samples = 200, batch_size = 1):
    X_test = MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),transforms.ToTensor()]))

    idx = np.where(X_test.targets == 1)[0][:n_samples]
    #    np.where(X_test.targets == 1)[0][:n_samples])
    # idx = np.stack([np.where(X_test.targets == i)[0][:n_samples] for i in range(10)], axis=1)
    # idx = idx.reshape(-1)

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
    return test_loader


#def train_dataset(n_samples = 200, batch_size = 1):
#    X_train = FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#
#    # # Leaving only labels 0 and 1 
#    idx = np.where(X_train.targets == 0)[0][:n_samples]
#    # idx = np.stack([np.where(X_train.targets == i)[0][:n_samples] for i in range(10)], axis=1)
#    # idx = idx.reshape(-1)
#
#    X_train.data = X_train.data[idx]
#    X_train.targets = X_train.targets[idx]
#
#    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=False, pin_memory=True)
#    return train_loader
#
#
#def test_dataset(n_samples = 200, batch_size = 1):
#    X_test = FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#
#    idx = np.where(X_test.targets == 0)[0][:n_samples]
#    # idx = np.stack([np.where(X_test.targets == i)[0][:n_samples] for i in range(10)], axis=1)
#    # idx = idx.reshape(-1)
#
#    X_test.data = X_test.data[idx]
#    X_test.targets = X_test.targets[idx]
#
#    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)
#    return test_loader
