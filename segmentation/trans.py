import cv2
import numpy as np


def crop_margin(img: cv2.Mat, eps: int = 4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img.shape[:2]

    def abs_bbox(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, x + w, y + h

    def is_lt_margin(bbox):
        l, t, _, _ = bbox
        il = l < eps
        it = t < eps
        return il or it

    def is_rb_margin(bbox):
        _, _, r, b = bbox
        ir = r > W - eps
        ib = b > H - eps
        return ir or ib

    bboxes = [abs_bbox(c) for c in contours]
    lt_margins = [bbox for bbox in bboxes if is_lt_margin(bbox)]
    rb_margins = [bbox for bbox in bboxes if is_rb_margin(bbox)]

    l = max(bbox[2] for bbox in lt_margins)
    t = max(bbox[3] for bbox in lt_margins)
    r = min(bbox[0] for bbox in rb_margins)
    b = min(bbox[1] for bbox in rb_margins)

    s = min(l, t)
    e = max(r, b)

    return img[s: e, s: e]

def clear_crop(img: cv2.Mat, eps: int = 2):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h, w), dtype=np.uint8)

    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        
        tl = np.any(pts[:, 0] <= eps)
        tt = np.any(pts[:, 1] <= eps)
        tr = np.any(pts[:, 0] >= w - eps)
        tb = np.any(pts[:, 1] >= h - eps)
        
        if not (tl or tt or tr or tb):
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    clean = np.full_like(img, 255)
    clean[mask == 255] = img[mask == 255]

    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    return clean

def bend(img: cv2.Mat, phase: float = 0):
    h, w = img.shape[:2]
    output = np.zeros(img.shape, dtype=np.uint8)

    amplitude = 20
    frequency = 2 * np.pi / w

    for x in range(w):
        shift = int(amplitude * np.sin(frequency * x + phase))
        for y in range(h):
            new_y = y + shift
            if 0 <= new_y < h:
                output[new_y, x] = img[y, x]

    #shade_strength = 0.15
    #shading = np.zeros((h, w), dtype=np.float32)
    #for x in range(w):
    #    shading[:, x] = 1 - shade_strength + shade_strength * np.sin(frequency * x)

    #shading = cv2.merge([shading, shading, shading])
    #output = np.clip(output * shading, 0, 255).astype(np.uint8)

    return output

def bend2(img: cv2.Mat, phase: float = 0.0, amplitude: int = 20, num_waves: float = 1.0):
    h, w = img.shape[:2]
    
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    frequency = 2 * np.pi * num_waves / w
    
    shift = amplitude * np.sin(frequency * X + phase)
    
    map_x = X.astype(np.float32)
    map_y = (Y - shift).astype(np.float32)
    
    output = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return output
