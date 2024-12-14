import torch
import torchvision
from torch.utils.data import DataLoader, random_split

# Function to get dataloaders with training and validation split
def get_dataloaders(batch_size, transform_train, transform_test, validation_split=0.2, random_seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    
    # Compute the size of the validation set
    num_train = len(trainset)
    num_val = int(num_train * validation_split)
    num_train = num_train - num_val
    
    # Split the dataset into training and validation sets
    train_data, val_data = random_split(trainset, [num_train, num_val])
    
    # Create DataLoaders for training, validation, and test sets
    trainloader = DataLoader(train_data, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    valloader = DataLoader(val_data, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    
    # Class labels for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, valloader, testloader, classes
    
