import torch
from torch.utils.data import Dataset

from functools import reduce
import lzma
import pickle
import random


class Base(Dataset):
    def __init__(self):
        super().__init__()
        self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label

    def split(self, dev: int):
        if dev >= len(self.data):
            raise ValueError('dev size must be at most the dataset size')
        left = MorseDataset()
        left.data = self.data[:dev]
        right = MorseDataset()
        right.data = self.data[dev:]
        return left, right


class MorseDataset(Base):
    @staticmethod
    def from_file(filename: str, dot_label=0, line_label=1):
        with lzma.open(filename, 'rb') as f:
            data = pickle.load(f)
        dots = [(torch.from_numpy(img), dot_label) for img in data['dots']]
        lines = [(torch.from_numpy(img), line_label) for img in data['lines']]
        data = [[dot, line] for dot, line in zip(dots, lines)]
        dataset = MorseDataset()
        dataset.data = reduce(lambda x, y: x + y, data, [])
        return dataset

    def __init__(self):
        super().__init__()


class MorseSequenceDataset(Base):
    def __init__(self, morse_dataset: MorseDataset, n: int, min_length: int = 1, max_length: int = 20, seed=None):
        super().__init__()
        if seed is not None:
            random.seed(seed)
        self.data = []
        for _ in range(n):
            length = random.randint(min_length, max_length)
            sequence = random.sample(morse_dataset.data, length)
            imgs = list(map(lambda x: x[0], sequence))
            labels = list(map(lambda x: x[1], sequence))
            self.data.append((torch.stack(imgs), labels, length))

    def __getitem__(self, idx):
        img, label, length = self.data[idx]
        return img, label, length
