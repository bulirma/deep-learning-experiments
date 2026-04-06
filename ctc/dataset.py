from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import closing, footprint_rectangle
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

import argparse
from datetime import datetime
import lzma
import pickle
import os
import sys


argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=None, help='seed (default: None)')
argparser.add_argument('--dataset', type=str, default='sequence', help='data to generate: symbols|sequence (default: sequence)')
argparser.add_argument('--min_len', type=int, default=1, help='minimal sequence length (used with sequence dataset, default: 1)')
argparser.add_argument('--max_len', type=int, default=20, help='maximal sequence length (used with sequence dataset, default: 20)')
argparser.add_argument('--seq_size', type=int, default=None, help='sequence dataset size')
argparser.add_argument('--dot_size', type=int, default=None, help='number of dots in symbols dataset')
argparser.add_argument('--line_size', type=int, default=None, help='number of lines in symbols dataset')
argparser.add_argument('--use_dataset', type=str, default=None, help='symbols dataset to generate sequence database')


class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def plt_show(img: np.array, title: str = None):
    plt.imshow(img, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def uniform_merge(a: list, b: list) -> list:
    la = len(a)
    lb = len(b)
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    if lb == 0:
        return a
    t = la + lb
    ratio = la / lb
    ai = 0
    bi = 0
    i = 0
    r = []
    while i < t:
        if (ai + 1) / (bi + 1) > ratio:
            r.append(b[bi])
            bi += 1
        else:
            r.append(a[ai])
            ai += 1
        i += 1
    return r

def load_morse_symbol_dataset(filename: str, split: int = 0, transform=None):
    def convert(img):
        img = torch.from_numpy(img).unsqueeze(0).float() / 255
        if transform is not None:
            img = transform(img)
        return img

    with lzma.open(filename, 'rb') as f:
        data = pickle.load(f)
    dots = [convert(img) for img in data['dots']]
    lines = [convert(img) for img in data['lines']]
    ld = len(dots)
    ll = len(lines)
    if split < 0:
        raise ValueError('split must be non-negative')
    if split > ld + ll:
        raise ValueError('split is greater than dataset size')
    dot_targets = [0] * ld
    line_targets = [1] * ll
    data = uniform_merge(dots, lines)
    targets = uniform_merge(dot_targets, line_targets)
    if split == 0:
        return SimpleDataset(data, targets)
    left_data = data[: -split]
    left_targets = targets[: -split]
    right_data = data[len(data) - split:]
    right_targets = targets[len(targets) - split:]
    return SimpleDataset(left_data, left_targets), SimpleDataset(right_data, right_targets)


def load_morse_sequence_dataset(filename: str, split: int = 0, transform=None):
    def convert(img):
        img = torch.from_numpy(img).unsqueeze(0).float() / 255
        if transform is not None:
            img = transform(img)
        return img

    with lzma.open(filename, 'rb') as f:
        data = pickle.load(f)
    seqs = [convert(img) for img in data['seqs']]
    labels = [torch.tensor(label, dtype=torch.uint8) for label in data['labels']]
    count = len(seqs)
    if split < 0:
        raise ValueError('split must be non-negative')
    if split > count:
        raise ValueError('split is greater than dataset size')
    if split == 0:
        return SimpleDataset(seqs, labels)
    left_data = seqs[: -split]
    left_targets = labels[: -split]
    right_data = seqs[len(seqs) - split:]
    right_targets = labels[len(labels) - split:]
    return SimpleDataset(left_data, left_targets), SimpleDataset(right_data, right_targets)

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

def gen_dot(
    img_shape: tuple,
    center_x_range: tuple = (0.2, 0.8),
    center_y_range: tuple = (0.2, 0.8),
    radius_range: tuple = (5, 15),
    irregularity: float = 0.15,
    num_blobs: int = 1
) -> np.array:
    width, height = img_shape
    
    img = np.zeros((height, width), dtype=np.float64)
    
    center_x = np.random.uniform(*center_x_range) * width
    center_y = np.random.uniform(*center_y_range) * height
    base_radius = np.random.uniform(*radius_range)
    
    num_angles = max(24, int(base_radius * 6))
    radius_variations = np.random.normal(1, irregularity, num_angles)
    radius_variations = np.clip(radius_variations, 0.6, 1.4)
    
    for row in range(height):
        for col in range(width):
            dx = col - center_x
            dy = row - center_y
            angle = np.arctan2(dy, dx)
            angle_idx = int((angle / (2 * np.pi) + 1) * num_angles / 2) % num_angles
            r_at_angle = base_radius * radius_variations[angle_idx]
            
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= r_at_angle:
                img[row, col] = 255
    
    img = img.astype(np.uint8)
    return img

def morph_line(img: np.array) -> np.array:
    return closing(img, footprint_rectangle((3, 3)))

def gen_line(
    img_shape: tuple,
    side_cut_range: tuple,
    amplitude_range: tuple = (2, 10),
    frequency_range: tuple = (0.01, 0.05),
    thickness: int = 3,
    noise: float = 0.1
) -> np.array:
    width, height = img_shape
    
    img = np.zeros((height, width), dtype=np.float64)
    
    base_y = np.random.uniform(height * 0.4, height * 0.6)
    amplitude = np.random.uniform(*amplitude_range)
    frequency = np.random.uniform(*frequency_range)
    phase = np.random.uniform(0, 2 * np.pi)
    curve_thickness = np.random.randint(1, thickness + 1)
    side_cut = np.random.randint(*side_cut_range)
    
    x = np.arange(width)
    y = base_y + amplitude * np.sin(frequency * x + phase)
    y += np.random.normal(0, noise * amplitude, width)
    
    for col in range(side_cut, width - side_cut):
        row = int(round(y[col]))
        for t in range(-curve_thickness // 2, curve_thickness // 2 + 1):
            if 0 <= row + t < height:
                img[row + t, col] = 255
    
    img = img.astype(np.uint8)
    return img

def get_spaces(img: np.array) -> list:
    spaces = []
    start = 0
    end = 0
    while True:
        c = end if start < end else start
        if c >= img.shape[1]:
            break
        space_column = np.all(img[:, c] == 0)
        if space_column:
            if start < end:
                end += 1
            else:
                end = start + 1
        else:
            if start < end:
                spaces.append((start, end))
                start = end + 1
            else:
                start += 1
    if start < end:
        spaces.append((start, end))
    return spaces

def cut_sides(img: np.array) -> np.array:
    preserve_cols = np.array([True] * img.shape[1])
    spaces = get_spaces(img)
    spaces = (spaces[0], spaces[-1]) if len(spaces) > 1 else spaces
    for start, end in spaces:
        cols = end - start
        offset = np.random.randint(cols // 2, cols)
        remove_cols = np.arange(start, start + offset)
        preserve_cols[remove_cols] = False
    return img[:, preserve_cols]

def gen_dots_and_lines(num_dots: int, num_lines: int) -> (list, list):
    dots = []
    lines = []
    with tqdm(range(num_dots + num_lines)) as pbar:
        for i in pbar:
            if i < num_dots:
                img = gen_dot(
                    (28, 28),
                    center_x_range=(0.25, 0.75),
                    center_y_range=(0.25, 0.75),
                    radius_range=(1, 3),
                    irregularity=0.10
                )
                dots.append(img)
            else:
                img = gen_line(
                    (28, 28),
                    side_cut_range=(2, 7),
                    amplitude_range=(2, 6),
                    frequency_range=(0.01, 0.03),
                    thickness=3,
                    noise=0.1
                )
                img = morph_line(img)
                lines.append(img)
    return dots, lines

def create_symbol_dataset(dots: int, lines: int) -> dict:
    dots, lines = gen_dots_and_lines(dots, lines)
    dataset = {
        'dots': dots,
        'lines': lines
    }
    return dataset

def create_sequence_dataset(symbol_dataset: dict, size: int, min_seq_len: int, max_seq_len: int):
    dots_len = len(symbol_dataset['dots'])

    def get_label(idx: int) -> int:
        nonlocal dots_len
        return 0 if idx < dots_len else 1

    symbols = np.array(symbol_dataset['dots'] + symbol_dataset['lines'], dtype=np.uint8)
    idxs = np.arange(symbols.shape[0])
    np.random.shuffle(idxs)

    seqs = []
    labels = []

    with tqdm(range(size)) as pbar:
        for _ in pbar:
            seq_idxs = np.random.choice(idxs, size=np.random.randint(min_seq_len, max_seq_len), replace=True)
            seq_symbols = symbols[seq_idxs]
            label = [get_label(idx) for idx in seq_idxs]
            cut_seq_symbols = []
            for i in range(len(label)):
                cut_seq_symbols.append(seq_symbols[i] if label[i] == 1 else cut_sides(seq_symbols[i]))
            seq = np.hstack(cut_seq_symbols)
            seqs.append(seq)
            labels.append(label)

    dataset = {
        'seqs': seqs,
        'labels': labels
    }
    return dataset

def main(args: argparse.Namespace):
    if args.seed is not None:
        np.random.seed(args.seed)

    if args.dataset == 'symbols' or (args.dataset == 'sequence' and args.use_dataset is None):
        if args.dot_size is not None and args.line_size is not None:
            dot_size = args.dot_size
            line_size = args.line_size
        elif args.dot_size is not None:
            dot_size = args.dot_size
            line_size = args.dot_size
        elif args.line_size is not None:
            dot_size = args.line_size
            line_size = args.line_size
        else:
            print('at least one of the arguemnts need to be specified: dot_size, line_size', file=sys.stderr)
            exit(1)
    elif args.dataset == 'sequence':
        if args.seq_size is None:
            print('seq_size must be specified', file=sys.stderr)
            exit(1)
        if not os.path.exists(args.use_dataset):
            print('symbols dataset file does not exist', file=sys.stderr)
            exit(1)

    if args.dataset == 'symbols':
        size_name_part = f'd{dot_size}l{line_size}'
        dataset = create_symbol_dataset(dot_size, line_size)
    elif args.dataset == 'sequence':
        size_name_part = f'n{args.seq_size}'
        if args.use_dataset is None:
            symbols = create_symbol_dataset(dot_size, line_size)
        else:
            with lzma.open(args.use_dataset, 'rb') as f:
                symbols = pickle.load(f)
        dataset = create_sequence_dataset(symbols, args.seq_size, args.min_len, args.max_len)
    else:
        print('unknown dataset', file=sys.stderr)
        exit(1)

    dt_name_part = f'd{datetime.now().strftime("%d%H%M%S")}'
    seed_name_part = '' if args.seed is None else f's{args.seed}'
    dataset_name = f'{args.dataset}_{size_name_part}{seed_name_part}{dt_name_part}.pklz'

    print('saving dataset (might take a while)...')
    with lzma.open(dataset_name, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main(argparser.parse_args(sys.argv[1:]))
