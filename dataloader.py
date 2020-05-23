from torchvision import datasets
import torch
from torchvision import transforms


def load_cifar(args):

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                          transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False,
                          transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)

    return train_loader, test_loader
