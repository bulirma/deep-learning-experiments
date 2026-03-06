import argparse
from datetime import datetime
import lzma
import os
import sys
import pickle
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EMNIST
import torchvision.transforms.v2 as transforms

import models


argparser = argparse.ArgumentParser()
argparser.add_argument('--dssrc', default=os.path.join(os.path.dirname(__file__), 'emnist'), type=str)
argparser.add_argument('--dataset', default='byclass', type=str, help='byclass, bymerge, balanced, letters, digits, mnist')
argparser.add_argument('--model_dir', default='models', type=str, help='directory to save models')
argparser.add_argument('--seed', default=42, type=int, help='randomization seed')
argparser.add_argument('--validation_set_size', default=0, type=int, help='size of validation set')
argparser.add_argument('--augment', action='store_true', help='enable data augmentation')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_emnist_train_data_statistics(root: str, split: str) -> (float, float):
    transform = transforms.Compose((
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ))
    train_raw = EMNIST(root=root, split=split, train=True, download=True, transform=transform)
    data = train_raw.data.float() / 255
    mean = (data.mean().item(),)
    std = (data.std().item(),)
    return mean, std

def get_emnist_loaders(root: str, split:str, batch_size: int, valid_size: int, seed: int, augment_transform=None) -> (DataLoader, DataLoader, Optional[DataLoader]):
    """
    :return: (train loader, dev loader, Opt[valid loader])
    """
    mean, std = get_emnist_train_data_statistics(root, split)
    transform = transforms.Compose((
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std)
    ))

    train = EMNIST(
        root=root, split=split, train=True, download=True, transform=transform if augment_transform is None else augment_transform)
    dev = EMNIST(root=root, split=split, train=False, download=True, transform=transform)

    dev_loader = DataLoader(dataset=dev, batch_size=batch_size * 2, shuffle=False)

    if valid_size == 0:
        train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
        return train_loader, dev_loader, None

    ts, vs = random_split(train, (train.size()[0] - valid_size, valid_size), generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(dataset=ts, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=vs, batch_size=batch_size * 2, shuffle=False)
    return train_loader, dev_loader, valid_loader


def main(args: argparse.Namespace):
    emnist_dir = args.dssrc
    batch_size = 100

    if args.augment:
        mean, std = get_emnist_train_data_statistics(emnist_dir, args.dataset)
        augment = transforms.Compose((
            transforms.RandomAffine(
                degrees=12,
                translate=(0.12, 0.12),
                scale=(0.88, 1.12),
                shear=10,
                fill=0
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value=0
            )
        ))
        loaders = get_emnist_loaders(emnist_dir, args.dataset, batch_size, args.validation_set_size, args.seed, augment)
    else:
        loaders = get_emnist_loaders(emnist_dir, args.dataset, batch_size, args.validation_set_size, args.seed)

    train_loader, test_loader, validation_loader = loaders

    steps_per_epoch = len(train_loader)
    learning_rate = 0.001
    weight_decay = 1e-4
    epochs = 15
    model_name = None

    if args.dataset == 'byclass':
        model = models.setup_byclass_cnn_model(62, learning_rate, weight_decay, args.validation_set_size > 0)
    elif args.dataset == 'mnist':
        model = models.setup_mnist_cnn_model(10, learning_rate, weight_decay, epochs, steps_per_epoch)
    else:
        raise 'dataset not implemented'
    
    logs = model.fit(epochs, train_loader, validation_loader)
    result = model.evaluate(test_loader)

    model_dir = os.path.join(os.path.dirname(__file__), args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    name = '' if model_name is None else f'{model_name}_'
    time = datetime.now().strftime('%d%H%M%S')
    model_fn = os.path.join(model_dir, f'{name}{args.dataset}_{time}.model')

    model_record = {
        'model': model,
        'train_logs': logs,
        'evaluation': result
    }
    with lzma.open(model_fn, 'wb') as mf:
        pickle.dump(model_record, mf)


if __name__ == '__main__':
    main(argparser.parse_args(sys.argv[1:]))
