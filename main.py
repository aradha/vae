import dataloader as dl
import options_parser as op
import train_network as trainer

def main(args):
    train_loader, test_loader = dl.load_cifar(args)
    trainer.train_network(train_loader, test_loader)

args = op.setup_args()
if __name__ == "__main__":
    main(args)
