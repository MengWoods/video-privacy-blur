import cv2
import numpy as np

def gaussian_inplace(frame, x1,y1,x2,y2, strength_divisor=8.0):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return
    kw = max(3, int((x2-x1)/strength_divisor) | 1)
    kh = max(3, int((y2-y1)/strength_divisor) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)

def pixelate_inplace(frame, x1,y1,x2,y2, blocks: int = 12):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return
    h, w = roi.shape[:2]
    fx = max(1, w // blocks)
    fy = max(1, h // blocks)
    small = cv2.resize(roi, (max(1,w//fx), max(1,h//fy)), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
