import torch
from torch.nn import functional as F
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
    def from_file(filename: str, normalize: bool = False, dot_label=0, line_label=1):
        with lzma.open(filename, 'rb') as f:
            data = pickle.load(f)
        dots = [(torch.from_numpy(img), dot_label) for img in data['dots']]
        lines = [(torch.from_numpy(img), line_label) for img in data['lines']]
        data = [[dot, line] for dot, line in zip(dots, lines)]
        dataset = MorseDataset()
        dataset.data = reduce(lambda x, y: x + y, data, [])
        if normalize:
            dataset.data = [(record[0].float() / 255, record[1]) for record in dataset.data]
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
            self.data.append((torch.cat(imgs, dim=1), torch.tensor(labels, dtype=torch.long), length))

    def __getitem__(self, idx):
        img, label, length = self.data[idx]
        return img, label, length


def pad_batch_images(imgs, pad_value=0):
    def pad(img, h, w):
        nonlocal pad_value
        b = h - img.size(0)
        r = w - img.size(1)
        return F.pad(img, (0, r, 0, b), mode='constant', value=pad_value)

    max_h = max(map(lambda img: img.size(0), imgs))
    max_w = max(map(lambda img: img.size(1), imgs))
    return torch.stack([pad(img, max_h, max_w) for img in imgs])

def collate(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    images_padded = pad_batch_images(imgs, pad_value=0)
    targets = torch.cat(targets).long()
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images_padded, targets, lengths
