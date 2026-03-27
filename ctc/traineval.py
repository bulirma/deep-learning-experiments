import torch
from torch.utils.data import DataLoader

import argparse
from datetime import datetime
import os
import sys

from datasets import MorseDataset, MorseSequenceDataset
import models


argparser = argparse.ArgumentParser()
argparser.add_argument('--model_dir', default='models', type=str, help='directory to save models')
argparser.add_argument('--seed', default=42, type=int, help='randomization seed')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args: argparse.Namespace):
    batch_size = 100
    learning_rate = 0.001
    weight_decay = 1e-4
    epochs = 15

    dataset = MorseDataset.from_file('morse-dataset.pklz')
    train, test = dataset.split(int(len(dataset) * 0.8))
    train = MorseSequenceDataset(train, 8000, 1, 20, args.seed)
    test = MorseSequenceDataset(test, 2000, 1, 20, args.seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=models.collate)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=models.collate)

    model = models.crnn_ctc_model(learning_rate, weight_decay)
    logs = model.fit(epochs, train_loader)
    result = model.evaluate(test_loader)

    model_record = {
        'model_state_dict': model.state_dict(),
        'train_logs': logs,
        'evaluation': result

    }
    model_dir = os.path.join(os.path.dirname(__file__), args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    time = datetime.now().strftime('%d%H%M%S')
    model_fn = os.path.join(model_dir, f'{time}.model')
    torch.save(model_record, model_fn)


if __name__ == '__main__':
    main(argparser.parse_args(sys.argv[1:]))
