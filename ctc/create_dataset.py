from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import closing, footprint_rectangle
from tqdm import tqdm

import lzma
import pickle


def draw(img: np.array):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def gen_dot(
    img_shape: tuple,
    center_x_range: tuple = (0.2, 0.8),
    center_y_range: tuple = (0.2, 0.8),
    radius_range: tuple = (5, 15),
    irregularity: float = 0.15,
    num_blobs: int = 1,
    seed: int = None
) -> np.array:
    if seed is not None:
        np.random.seed(seed)

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
    noise: float = 0.1,
    seed: int = None
) -> np.array:
    width, height = img_shape

    if seed is not None:
        np.random.seed(seed)
    
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

def main():
    dots = []
    lines = []
    with tqdm(range(2500)) as pbar:
        for _ in pbar:
            img = gen_dot(
                (28, 28),
                center_x_range=(0.25, 0.75),
                center_y_range=(0.25, 0.75),
                radius_range=(1, 4),
                irregularity=0.10
            )
            dots.append(img)

            img = gen_line(
                (28, 28),
                side_cut_range=(2, 5),
                amplitude_range=(2, 6),
                frequency_range=(0.01, 0.03),
                thickness=3,
                noise=0.1,
            )
            img = morph_line(img)
            lines.append(img)
    dataset = {
        'dots': dots,
        'lines': lines
    }
    with lzma.open('morse-dataset.pklz', 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
