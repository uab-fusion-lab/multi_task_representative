import random

import torch
import torchvision

import torch
import torch.nn as nn
import torch.optim as optim
# from torchtext.datasets import IMDB
# from torchtext.data import Field, BucketIterator
from torch.utils.data import DataLoader, TensorDataset



def fetch_dataset():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    return train_data, test_data

def fetch_FashionMNIST_dataset():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.FashionMNIST(
        './data', train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.FashionMNIST(
        './data', train=False, download=True, transform=transform
    )

    return train_data, test_data
def fetch_emnist_dataset():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.EMNIST(
        './data', train=True, split='byclass', download=True, transform=transform
    )

    test_data = torchvision.datasets.EMNIST(
        './data', train=False, split='byclass', download=True, transform=transform
    )

    return train_data, test_data

def fetch_svhn_dataset():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.SVHN(root='./data', split='train', transform=transform, download=True)
    test_data = torchvision.datasets.SVHN(root='./data', split='test', transform=transform, download=True)
    # Create a list of indices
    indices = list(range(70000))

    # Create a subset of the training dataset
    train_data = torch.utils.data.Subset(train_data, indices)

    return train_data, test_data
# def fetch_IMDb_dataset():
#     # Define Fields
#     TEXT = Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True)
#     LABEL = Field(sequential=False, use_vocab=False)
#
#     # Load IMDb dataset
#     train_data, test_data = IMDB.splits(TEXT, LABEL)
#
#     # Build vocabulary
#     TEXT.build_vocab(train_data, max_size=5000, vectors="glove.6B.100d")
#     LABEL.build_vocab(train_data)
#
#     # Create iterators
#     train_iterator, test_iterator = BucketIterator.splits(
#         (train_data, test_data),
#         batch_size=32,
#         sort_within_batch=True
#     )
#
#     return train_data, test_data


def fetch_fashion_dataset():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    return train_data, test_data

def fetch_svhn_dataset():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.SVHN(root='./data', split='train', transform=transform, download=True)

    test_data = torchvision.datasets.SVHN(root='./data', split='test', transform=transform, download=True)

    # Create a list of indices for this size
    indices = list(range(70000))

    # Create a subset of the training dataset
    train_data = torch.utils.data.Subset(train_data, indices)
    return train_data, test_data



def fetch_cifar10_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the training dataset
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # Download and load the test dataset
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    return train_data, test_data

def fetch_cifar100_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the training dataset
    train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # Download and load the test dataset
    test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    return train_data, test_data


def data_to_tensor(data):
    """ Loads dataset to memory, applies transform"""
    loader = torch.utils.data.DataLoader(data, batch_size=len(data))
    img, label = next(iter(loader))
    return img, label


def iid_partition_loader(data, bsz=10, n_clients=10):
    """ partition the dataset into a dataloader for each client, iid style
    """
    m = len(data)
    assert m % n_clients == 0
    m_per_client = m // n_clients
    # assert m_per_client % bsz == 0
    list = [m_per_client for x in range(n_clients)]
    list[len(list)-1] += 32
    client_data = torch.utils.data.random_split(
        data,
        # list
        [m_per_client for x in range(n_clients)]
    )
    client_loader = [
        torch.utils.data.DataLoader(x, batch_size=bsz, shuffle=True)
        for x in client_data
    ]
    return client_loader


def noniid_partition_loader(
    data, bsz=10, m_per_shard=200, n_shards_per_client=1
):
    """ semi-pathological client sample partition
    1. sort examples by label, form shards of size 300 by grouping points
       successively
    2. each client is 2 random shards
    most clients will have 2 digits, at most 4
    """

    # load data into memory
    img, label = data_to_tensor(data)

    # sort
    idx = torch.argsort(label)
    img = img[idx]
    label = label[idx]

    # split into n_shards of size m_per_shard
    m = len(data)
    # assert m % m_per_shard == 0
    n_shards = m // m_per_shard
    shards_idx = [
        torch.arange(m_per_shard*i, m_per_shard*(i+1))
        for i in range(n_shards)
    ]
    random.shuffle(shards_idx)  # shuffle shards

    # pick shards to create a dataset for each client
    assert n_shards % n_shards_per_client == 0
    n_clients = n_shards // n_shards_per_client
    client_data = [
        torch.utils.data.TensorDataset(
            torch.cat([img[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)]),
            torch.cat([label[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)])
        )
        for i in range(n_clients)
    ]

    # make dataloaders
    client_loader = [
        torch.utils.data.DataLoader(x, batch_size=bsz, shuffle=True)
        for x in client_data
    ]
    return client_loader
