from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import argparse
from functools import reduce
import lzma
import os
import pickle
import random
import sys

from datasets import MorseDataset, MorseSequenceDataset


argparser = argparse.ArgumentParser()
argparser.add_argument('--model_dir', default='models', type=str, help='directory to save models')
argparser.add_argument('--seed', default=42, type=int, help='randomization seed')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args: argparse.Namespace):
    batch_size = 100

    dataset = MorseDataset.from_file('morse-dataset.pklz')
    train, test = dataset.split(int(len(dataset) * 0.8))
    train = MorseSequenceDataset(train, 8000, 1, 20, args.seed)
    test = MorseSequenceDataset(test, 2000, 1, 20, args.seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    #for i, (img, label) in enumerate(train):
    #    if i >= 10:
    #        break
    #    plt.imshow(np.hstack(img.numpy()), cmap='gray')
    #    plt.title(str(label))
    #    plt.axis('off')
    #    plt.tight_layout()
    #    plt.show()

    #train_loader, test_loader, validation_loader = loaders

    #steps_per_epoch = len(train_loader)
    #learning_rate = 0.001
    #weight_decay = 1e-4
    #epochs = 15
    #model_name = None

    #if args.dataset == 'byclass':
    #    model = models.setup_byclass_cnn_model(62, learning_rate, weight_decay, args.validation_set_size > 0)
    #elif args.dataset == 'mnist':
    #    model = models.setup_mnist_cnn_model(10, learning_rate, weight_decay, epochs, steps_per_epoch)
    #else:
    #    raise 'dataset not implemented'
    #
    #logs = model.fit(epochs, train_loader, validation_loader)
    #result = model.evaluate(test_loader)

    #model_dir = os.path.join(os.path.dirname(__file__), args.model_dir)
    #os.makedirs(model_dir, exist_ok=True)
    #name = '' if model_name is None else f'{model_name}_'
    #time = datetime.now().strftime('%d%H%M%S')
    #model_fn = os.path.join(model_dir, f'{name}{args.dataset}_{time}.model')

    #model_record = {
    #    'model_state_dict': model.state_dict(),
    #    'train_logs': logs,
    #    'evaluation': result
    #}
    #torch.save(model_record, model_fn)


if __name__ == '__main__':
    main(argparser.parse_args(sys.argv[1:]))
