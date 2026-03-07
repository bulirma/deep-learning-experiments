import random
import sys

from matplotlib import pyplot as plt
import pygame
import torch
from torchvision.datasets import EMNIST
import torchvision.transforms.v2 as transforms

import models
from models import Model
from traineval import DEVICE


class Canvas:
    _point_stroke = {
        (0, 0): 0
    }
    _small_stroke = {
        (-1, -1): 63,
        (-1, 1): 63,
        (1, -1): 63,
        (1, 1): 63,
        (-1, 0): 31,
        (0, -1): 31,
        (0, 1): 31,
        (1, 0): 31,
        (0, 0): 0
    }
    _erase_stroke = {
        (-1, -1): 255,
        (-1, 1): 255,
        (1, -1): 255,
        (1, 1): 255,
        (-1, 0): 255,
        (0, -1): 255,
        (0, 1): 255,
        (1, 0): 255,
        (0, 0): 255
    }

    def __init__(self, screen, x, y, c):
        self.screen = screen
        self.x = x + 1
        self.y = y + 1
        self.sc = 28
        self.c = c
        self.s = self.sc * c
        self.c = self.s // self.sc
        self._stroke = self._small_stroke
        self._eraser = False
        self.clear()

    def render(self):
        pygame.draw.rect(self.screen, (0, 0, 0), (self.x, self.y, self.s, self.s), 1)
        for xi in range(self.sc):
            sx = xi * self.c + self.x
            for yi in range(self.sc):
                if self.image[xi, yi] == 255:
                    continue
                sy = yi * self.c + self.y
                g = self.image[xi, yi]
                pygame.draw.rect(self.screen, (g, g, g), (sx, sy, self.c, self.c), 0)

    def _is_at(self, x, y):
        return 0 <= x < self.s and 0 <= y < self.s

    def is_at(self, ax, ay):
        x, y = ax - self.x, ay - self.y
        return self._is_at(x, y)

    def _apply_stroke(self, xi, yi):
        for (oxi, oyi), c in self._stroke.items():
            rxi, ryi = oxi + xi, oyi + yi
            if rxi < 0 or ryi < 0 or rxi >= self.sc or ryi >= self.sc:
                continue
            if self._eraser:
                self.image[rxi, ryi] |= c
            else:
                self.image[rxi, ryi] &= c

    def draw(self, ax, ay):
        x, y = ax - self.x, ay - self.y
        xi = x // self.c
        yi = y // self.c
        self._apply_stroke(xi, yi)

    def clear(self):
        self.image = torch.ones((self.sc, self.sc), dtype=torch.uint8) * 255

    def set_point_stroke(self):
        self._eraser = False
        self._stroke = self._point_stroke

    def set_small_stroke(self):
        self._eraser = False
        self._stroke = self._small_stroke

    def set_erase_stroke(self):
        self._eraser = True
        self._stroke = self._erase_stroke


def main():
    pygame.init()
    screen = pygame.display.set_mode((464, 464))
    pygame.display.set_caption('MNIST demo')
    clock = pygame.time.Clock()

    canvas = Canvas(screen, 8, 8, 16)
    model_record = torch.load('models/byclass_07200326.model', map_location=torch.device(DEVICE))
    model_state = model_record['model_state_dict']
    model = models.setup_byclass_cnn_model(62, 0.001, 1e-4, False)
    model.load_state_dict(model_state)
    classes = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghijklmnopqrstuvwxyz'

    running = True
    while running:
        screen.fill('white')
        clock.tick(60)

        canvas.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)
                if key == 'c':
                    canvas.clear()
                elif key == 'p':
                    image = (255 - canvas.image)
                    image = image.float() / 255.0
                    image = (image - 0.1307) / 0.3081
                    image = image.unsqueeze(0).unsqueeze(0)
                    print(classes[model.predict(image)])
                elif key == '1':
                    canvas.set_point_stroke()
                elif key == '2':
                    canvas.set_small_stroke()
                elif key == '3':
                    canvas.set_erase_stroke()

        left_pressed = pygame.mouse.get_pressed()[0]
        pos = pygame.mouse.get_pos()
        if left_pressed and canvas.is_at(*pos):
            canvas.draw(*pos)

        pygame.display.flip()

    pygame.quit()

def show_dataset(c: int = None):
    transform = transforms.Compose((
        transforms.ToImage(),
    ))
    train_raw = EMNIST(root='emnist', split='byclass', train=True, download=True, transform=transform)

    if c is None:
        class_indices = torch.arange(train_raw.data.size(0))
    else:
        class_indices = (train_raw.targets == c).nonzero().squeeze()

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        i = random.choice(class_indices)
        image = train_raw.data[i]
        target = train_raw.targets[i]

        ax.imshow(image.T, cmap='gray')
        ax.set_title(f'label: {target.item()}')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    #show_dataset()
    #show_dataset(int(sys.argv[1]))
