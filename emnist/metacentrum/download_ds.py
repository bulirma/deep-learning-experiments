import argparse
import os
import sys

from torchvision.datasets import EMNIST
import torchvision.transforms.v2 as transforms

argparser = argparse.ArgumentParser()
argparser.add_argument('--dstrg', default=os.path.join(os.path.dirname(__file__), 'emnist'), type=str)
argparser.add_argument('--dataset', default='byclass', type=str, help='byclass, bymerge, balanced, letters, digits, mnist')


if __name__ == '__main__':
    args = argparser.parse_args(sys.argv[1:])
    transform = transforms.Compose((transforms.ToImage(),))
    _ = EMNIST(root=args.dstrg, split=args.dataset, train=True, download=True, transform=transform)
    _ = EMNIST(root=args.dstrg, split=args.dataset, train=False, download=True, transform=transform)
