import lzma
import pickle

import pygame
import torch

from models import Model


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

    def __init__(self, screen, x, y, c):
        self.screen = screen
        self.x = x + 1
        self.y = y + 1
        self.sc = 28
        self.c = c
        self.s = self.sc * c
        self.c = self.s // self.sc
        self._stroke = self._small_stroke
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
            self.image[rxi, ryi] &= c

    def draw(self, ax, ay):
        x, y = ax - self.x, ay - self.y
        xi = x // self.c
        yi = y // self.c
        self._apply_stroke(xi, yi)

    def clear(self):
        self.image = torch.ones((self.sc, self.sc), dtype=torch.uint8) * 255


def main():
    pygame.init()
    screen = pygame.display.set_mode((464, 464))
    pygame.display.set_caption('MNIST demo')
    clock = pygame.time.Clock()

    canvas = Canvas(screen, 8, 8, 16)
    with lzma.open('models/no_augment.model', 'rb') as mf:
        model_record = pickle.load(mf)
    model = model_record['model']

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
                    image = (255 - canvas.image).T
                    image = image.float() / 255.0
                    image = (image - 0.1307) / 0.3081
                    image = image.unsqueeze(0).unsqueeze(0)
                    print(model.predict(image))

        left_pressed = pygame.mouse.get_pressed()[0]
        pos = pygame.mouse.get_pos()
        if left_pressed and canvas.is_at(*pos):
            canvas.draw(*pos)

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
