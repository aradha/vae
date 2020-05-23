import argparse


def setup_args():
    parser = argparse.ArgumentParser(description='CIFAR 10 VAE')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

    return parser.parse_args()
