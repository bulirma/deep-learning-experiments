import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import random

from trans import bend, bend2, crop_margin, clear_crop


def plt_show(title: str, img: cv2.Mat):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(title)
    plt.show()

def line_y_bounds(img: cv2.Mat):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def get_y(contour):
        _, y, _, h = cv2.boundingRect(contour)
        return y + h / 2

    contours = sorted(contours, key=get_y)

    lines = []
    current_line = [contours[0]]
    
    for c in contours[1:]:
        _, y_prev, _, h_prev = cv2.boundingRect(current_line[-1])
        _, y, _, h = cv2.boundingRect(c)
        if y - (y_prev + h_prev) < 20:
            current_line.append(c)
        else:
            lines.append(current_line)
            current_line = [c]
    lines.append(current_line)

    bounds = []
    for i, line_contours in enumerate(lines):
        points = np.vstack(line_contours).reshape(-1, 2)
        bbound = np.min(points[:, 1]) - 1
        tbound = np.max(points[:, 1]) + 1
        bounds.append((bbound, tbound))

    return bounds

def neume_x_bounds(img: cv2.Mat):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def get_x(contour):
        x, _, w, _ = cv2.boundingRect(contour)
        return x + w / 2

    contours = sorted(contours, key=get_x)

    neumes = []
    current_neume = [contours[0]]
    
    for c in contours[1:]:
        x_prev, _, w_prev, _ = cv2.boundingRect(current_neume[-1])
        x, _, _, _ = cv2.boundingRect(c)
        if x - (x_prev + w_prev) < 6:
            current_neume.append(c)
        else:
            neumes.append(current_neume)
            current_neume = [c]
    neumes.append(current_neume)

    bounds = []
    for i, neume in enumerate(neumes):
        points = np.vstack(neume).reshape(-1, 2)
        lbound = np.min(points[:, 0]) - 1
        rbound = np.max(points[:, 0]) + 1
        bounds.append((lbound, rbound))

    return bounds


def get_line_images(img: cv2.Mat):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    closing_kernel = np.ones((12, 88), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, closing_kernel)
    dilatation_kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(closed, dilatation_kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    line_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if h > 30 and w > 150 and area > 1000:
            line_contours.append((c, y))
    
    line_contours.sort(key=lambda x: x[1])

    line_imgs = []
    for i, (contour, _) in enumerate(line_contours):
        x, y, w, h = cv2.boundingRect(contour)
        l = x - 2
        r = x + w + 2
        t = y - 2
        b = y + h + 2

        #mask = np.zeros_like(gray)
        #cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        #line_mask = mask[t: b, l: r]
        line_img = img[t: b, l: r]
        #line_img[line_mask == 0] = 255

        line_imgs.append(line_img)

    return line_imgs

    
def main():
    img = cv2.imread('neumes.png')
    #bent = img
    bent = bend(img)
    bent = crop_margin(bent)
    lines = get_line_images(bent)
    line = clear_crop(lines[0])
    plt_show('line', line)
    neume_bounds = neume_x_bounds(line)
    for i, bounds in enumerate(neume_bounds):
        l, r = bounds
        neume_img = line[:, l - 1: r + 2]
        plt_show(f'neume {i + 1}', neume_img)
    #for line in lines:
    #    line = clear_crop(line)
    #    plt_show('line', line)
    #neume_bounds = neume_x_bounds(img[b - 1: t + 2, :])
    #for i, bounds in enumerate(neume_bounds):
    #    l, r = bounds
    #    neume_img = img[b - 1: t + 2, l - 1: r + 2]
    #    plt_show(f'neume {i + 1}', neume_img)


    #for line in lines:
    #    plt_show('line', line)
    #    test3(line)
    #plt_show('neumes', img)
    #line_bounds = line_y_bounds(img)
    #b, t = line_bounds[0]
    #neume_bounds = neume_x_bounds(img[b - 1: t + 2, :])
    #for i, bounds in enumerate(neume_bounds):
    #    l, r = bounds
    #    neume_img = img[b - 1: t + 2, l - 1: r + 2]
    #    plt_show(f'neume {i + 1}', neume_img)
    
    #line_bounds = line_y_bounds(bent)
    #print(line_bounds)
    #b, t = line_bounds[0]
    #line_img = bent[b - 1: t + 2, :]
    #plt_show('line', line_img)
    #neume_bounds = neume_x_bounds(bent[b - 1: t + 2, :])
    #for i, bounds in enumerate(neume_bounds):
    #    l, r = bounds
    #    neume_img = bent[b - 1: t + 2, l - 1: r + 2]
    #    plt_show(f'neume {i + 1}', neume_img)


if __name__ == "__main__":
    main()
