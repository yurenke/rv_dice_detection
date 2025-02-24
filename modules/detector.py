import os
import torch
import cv2
from ultralytics import YOLO

class DiceDetector():
    def __init__(self, config) -> None:
        self.load_model()
        self.config = config['detector']['detect_area']
    
    def load_model(self):
        model_path  = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', 'yolo', 'dice.pt')
        self.model = YOLO(model_path).to('cpu')

    def detect(self, frame):
        dice_area = frame[self.config['ymin']:self.config['ymax'], self.config['xmin']:self.config['xmax']]
        # 偵測骰子
        results = self.model(dice_area, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()  # (xmin, ymin, xmax, ymax, confidence, class)
        # confidence必須夠高，否則視為仍模糊在滾動無法清楚辨識
        dice_detections = [d for d in detections if d[4] >= 0.9]
        return dice_detections