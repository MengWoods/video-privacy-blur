from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

class PlateDetectorYOLO:
    def __init__(self, weights: str, device: str="cpu", conf: float=0.35, imgsz: int=960):
        self.model = YOLO(weights)  # loads local file
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frame):
        res = self.model(frame, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        return [tuple(map(int, xy)) for xy in res.boxes.xyxy.cpu().numpy()]


class FaceDetectorHaar:
    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)

    def __call__(self, frame) -> List[Tuple[int,int,int,int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

class FaceDetectorYOLO:
    def __init__(self, weights: str = "yolov8n-face.pt", device:str="cpu", conf:float=0.35, imgsz:int=960):
        self.model = YOLO(weights)  # local file or hub id
        self.device = device
        self.conf = max(0.25, conf)
        self.imgsz = imgsz

    def __call__(self, frame) -> List[Tuple[int,int,int,int]]:
        res = self.model(frame, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes)==0: return []
        return [tuple(map(int, xy)) for xy in res.boxes.xyxy.cpu().numpy()]
