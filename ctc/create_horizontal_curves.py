import numpy as np
import matplotlib.pyplot as plt


def create_horizontal_curve(
    height: int,
    width: int,
    num_curves: int = 5,
    amplitude_range: tuple = (2, 10),
    frequency_range: tuple = (0.01, 0.05),
    thickness: int = 3,
    noise: float = 0.1,
    seed: int = None
) -> np.array:
    if seed is not None:
        np.random.seed(seed)
    
    img = np.zeros((height, width), dtype=np.float64)
    
    for _ in range(num_curves):
        base_y = np.random.uniform(height * 0.2, height * 0.8)
        amplitude = np.random.uniform(*amplitude_range)
        frequency = np.random.uniform(*frequency_range)
        phase = np.random.uniform(0, 2 * np.pi)
        curve_thickness = np.random.randint(1, thickness + 1)
        
        x = np.arange(width)
        y = base_y + amplitude * np.sin(frequency * x + phase)
        y += np.random.normal(0, noise * amplitude, width)
        
        for col in range(width):
            row = int(round(y[col]))
            for t in range(-curve_thickness // 2, curve_thickness // 2 + 1):
                if 0 <= row + t < height:
                    img[row + t, col] = 255
    
    img = img.astype(np.uint8)
    return img


def create_smooth_horizontal_curve(
    height: int,
    width: int,
    num_curves: int = 5,
    waviness: float = 0.02,
    amplitude: float = 15,
    thickness: int = 2,
    seed: int = None
) -> np.array:
    if seed is not None:
        np.random.seed(seed)
    
    img = np.zeros((height, width), dtype=np.float64)
    
    for _ in range(num_curves):
        base_y = np.random.uniform(height * 0.25, height * 0.75)
        
        x = np.arange(width)
        noise = np.cumsum(np.random.normal(0, waviness, width))
        y = base_y + amplitude * np.sin(waviness * 5 * x + noise)
        
        y_clipped = np.clip(y, 0, height - 1)
        
        for col in range(width):
            row = int(round(y_clipped[col]))
            for t in range(-thickness // 2, thickness // 2 + 1):
                if 0 <= row + t < height:
                    dist = abs(row + t - y_clipped[col])
                    intensity = max(0, 255 * (1 - dist / (thickness + 1)))
                    img[row + t, col] = max(img[row + t, col], intensity)
    
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def visualize(img: np.array, title: str = "Horizontal Curves"):
    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    #img1 = create_horizontal_curve(
    #    height=100,
    #    width=300,
    #    num_curves=5,
    #    amplitude_range=(5, 15),
    #    frequency_range=(0.02, 0.08),
    #    thickness=3,
    #    noise=0.2,
    #    seed=42
    #)
    #visualize(img1, "Nearly Straight Horizontal Curves (Sine-based)")
    #
    #img2 = create_smooth_horizontal_curve(
    #    height=100,
    #    width=300,
    #    num_curves=4,
    #    waviness=0.03,
    #    amplitude=20,
    #    thickness=2,
    #    seed=123
    #)
    #visualize(img2, "Nearly Straight Horizontal Curves (Random Walk)")
    
    img3 = create_horizontal_curve(
        height=200,
        width=400,
        num_curves=1,
        amplitude_range=(2, 6),
        frequency_range=(0.01, 0.03),
        thickness=8,
        noise=0.1,
        #seed=456
    )
    visualize(img3, "Subtle Nearly Straight Horizontal Curves")


if __name__ == '__main__':
    main()
