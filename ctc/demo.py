from matplotlib import pyplot as plt
import numpy as np
import pygame
import torch

import argparse
import os
import sys

import models
from models import CTCModel
from traineval import DEVICE
from dataset import load_morse_symbol_dataset, load_morse_sequence_dataset


argparser = argparse.ArgumentParser()
argparser.add_argument('--predict', type=str, default=None, help='model')
argparser.add_argument('--show_dataset', type=str, default=None, help='dataset')


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

    def __init__(self, screen, x, y, c, scw, sch):
        self.screen = screen
        self.x = x + 1
        self.y = y + 1
        self.scw = scw
        self.sch = sch
        self.sw = self.scw * c
        self.sh = self.sch * c
        self.c = c
        self._stroke = self._small_stroke
        self._eraser = False
        self.clear()

    def render(self):
        pygame.draw.rect(self.screen, (0, 0, 0), (self.x, self.y, self.sw, self.sh), 1)
        for xi in range(self.scw):
            sx = xi * self.c + self.x
            for yi in range(self.sch):
                if self.image[xi, yi] == 255:
                    continue
                sy = yi * self.c + self.y
                g = self.image[xi, yi]
                pygame.draw.rect(self.screen, (g, g, g), (sx, sy, self.c, self.c), 0)

    def _is_at(self, x, y):
        return 0 <= x < self.sw and 0 <= y < self.sh

    def is_at(self, ax, ay):
        x, y = ax - self.x, ay - self.y
        return self._is_at(x, y)

    def _apply_stroke(self, xi, yi):
        for (oxi, oyi), c in self._stroke.items():
            rxi, ryi = oxi + xi, oyi + yi
            if rxi < 0 or ryi < 0 or rxi >= self.scw or ryi >= self.sch:
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
        self.image = torch.ones((self.scw, self.sch), dtype=torch.uint8) * 255

    def set_point_stroke(self):
        self._eraser = False
        self._stroke = self._point_stroke

    def set_small_stroke(self):
        self._eraser = False
        self._stroke = self._small_stroke

    def set_erase_stroke(self):
        self._eraser = True
        self._stroke = self._erase_stroke


def plt_show(img: np.array, title: str = None):
    plt.imshow(img, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def predict(model_path: str):
    GRID_W = 140
    GRID_H = 28
    GRID_C = 8
    GRID_LM = 8
    GRID_TM = GRID_LM
    CANVAS_W = GRID_C * GRID_W + 2 * GRID_LM
    CANVAS_H = GRID_C * GRID_H + 2 * GRID_TM

    pygame.init()
    screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
    pygame.display.set_caption('MNIST demo')
    clock = pygame.time.Clock()

    canvas = Canvas(screen, GRID_LM, GRID_TM, GRID_C, GRID_W, GRID_H)
    model_record = torch.load(model_path, map_location=torch.device(DEVICE))
    model_state = model_record['model_state_dict']
    learning_rate = 0.001
    weight_decay = 1e-4
    model = models.crnn_ctc_model(learning_rate, weight_decay)
    model.load_state_dict(model_state)
    #classes = '.-'

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
                    image = image.T
                    image = image.unsqueeze(0)
                    prediction = model.predict(image)
                    print(prediction)
                    #print(''.join(classes[ci] for ci in decoded[0]))
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

#def show_results():
#    model_record = torch.load('models/02052910.model', map_location=torch.device(DEVICE))
#    #model_state = model_record['model_state_dict']
#    #learning_rate = 0.001
#    #weight_decay = 1e-4
#    #model = models.crnn_ctc_model(learning_rate, weight_decay)
#    #model.load_state_dict(model_state)
#    logs = model_record['train_logs']
#    evaluation = model_record['evaluation']
#    print(logs)
#    print(evaluation)

def show_dataset(dataset_fn: str):
    if dataset_fn.startswith('symbols'):
        dataset = load_morse_symbol_dataset(dataset_fn)
    else:
        dataset = load_morse_sequence_dataset(dataset_fn)
    for i in range(30):
        img, label = dataset[i]
        plt_show(img, str(label))

def main(args: argparse.Namespace):
    if args.predict is not None:
        if not os.path.exists(args.predict):
            print('model file does not exist', file=sys.stderr)
            exit(1)
        predict(args.predict)
    if args.show_dataset is not None:
        if not os.path.exists(args.show_dataset):
            print('dataset file does not exist', file=sys.stderr)
            exit(1)
        show_dataset(args.show_dataset)


if __name__ == '__main__':
    main(argparser.parse_args(sys.argv[1:]))
