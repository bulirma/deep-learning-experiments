from matplotlib import pyplot as plt
import numpy as np
import random
from skimage.morphology import closing, footprint_rectangle, opening, disk, erosion


def draw(img: np.array):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def threshold(img: np.array, t: int) -> np.array:
    add = (255 - img) / 3
    img = np.where(img > t, img + add, 0)
    return img

def threshold_del(img: np.array, t: int) -> np.array:
    cond_range_add = (255 - t) / 4
    min_val = np.where(img != 0, img, 255).min()
    cond = np.logical_and(img != 0, img <= min_val + cond_range_add)
    rnd = np.random.random(size=img.shape)
    cond = np.logical_and(cond, rnd > 0.5)
    select = np.where(cond, img, 0)
    img = np.where(img > min_val + cond_range_add, img, select)
    return img

def morph(img: np.array) -> np.array:
    img = closing(img, footprint_rectangle((3, 3)))
    img = opening(img, disk(2))
    return img

def gen_dot(img_shape: tuple, shift: tuple = (0, 0), t: int = 180) -> np.array:
    min_val = 0
    max_val = 255
    half = (max_val - min_val) / 2
    step_x = (max_val - min_val) / img_shape[0]
    shift_x = shift[0] * step_x
    step_y = (max_val - min_val) / img_shape[1]
    shift_y = shift[1] * step_y
    x = np.linspace(min_val - shift_x, max_val - shift_x, img_shape[0])
    y = np.linspace(min_val - shift_y, max_val - shift_y, img_shape[1])
    X, Y = np.meshgrid(x, y, indexing='xy')
    X_b = max_val - np.abs(half - X)
    Y_b = max_val - np.abs(half - Y)
    Z = (X_b + Y_b) ** 4
    real_max = Z.max()
    Z *= 255 / real_max
    img = threshold(Z, t)
    img = threshold_del(img, t)
    if np.sum(img != 0) > 10:
        img = morph(img)
    img = img.astype(np.uint8)
    return img

#def gen_line(img_shape: tuple, shift: tuple = (0, 0), t: int = 140) -> np.array:
#    min_val = 0
#    max_val = 255
#    half = (max_val - min_val) / 2
#    step_x = (max_val - min_val) / img_shape[0]
#    shift_x = shift[0] * step_x
#    step_y = (max_val - min_val) / img_shape[1]
#    shift_y = shift[1] * step_y
#    x = np.linspace(min_val - shift_x, max_val - shift_x, img_shape[0])
#    y = np.linspace(min_val - shift_y, max_val - shift_y, img_shape[1])
#    X, Y = np.meshgrid(x, y, indexing='xy')
#    Y_b = max_val - np.abs(half - Y)
#    Z = Y_b ** 4
#    real_max = Z.max()
#    Z *= 255 / real_max
#    rnd = random.random()
#    if rnd > 0.75:
#        Z = np.vstack([
#            Z[1: img_shape[1] // 2],
#            Z[img_shape[1] // 2 + 1:],
#            Z[img_shape[1] - 1]
#        ])
#    elif rnd > 0.5:
#        Z = np.vstack([
#            Z[0],
#            Z[0: img_shape[1] // 2],
#            Z[img_shape[1] // 2 + 1:]
#        ])
#    M = np.ones_like(Z) * 255
#    n = random.randint(3, 6)
#    for i in range(n):
#        step = 255 // n
#        M[:, i] = i * step
#        M[:, img_shape[0] - i - 1] = i * step
#    Z += M
#    Z **= 4
#    real_max = Z.max()
#    Z *= 255 / real_max
#    img = Z
#    img = threshold(Z, t)
#    img = img.astype(np.uint8)
#    return img

def gen_line(img_shape: tuple, shift: tuple = (0, 0), t: int = 140) -> np.array:
    pass

def main():
    a = 28
    shift = (2, 1)
    img = gen_dot((a, a), shift, 180)
    draw(img)

    a = 28
    shift = (0, 0)
    img = gen_line((a, a), shift)
    #print(img)
    #print(img.min(), img.max())
    draw(img)


if __name__ == '__main__':
    main()
