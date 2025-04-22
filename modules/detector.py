import os
import torch
import cv2
from ultralytics import YOLO

class DiceDetector():
    def __init__(self) -> None:
        self.load_model()
    
    def load_model(self):
        model_path  = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', 'yolo', 'dice.pt')
        self.model = YOLO(model_path).to('cpu')

    def detect(self, img):
        # 偵測骰子
        results = self.model(img, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()  # (xmin, ymin, xmax, ymax, confidence, class)
        # confidence必須夠高，否則視為仍模糊在滾動無法清楚辨識
        dice_detections = [d for d in detections if d[4] >= 0.9]
        return dice_detections
    
class DiceCupBaseDetector():
    def __init__(self, config) -> None:
        self.config = config['detector']
        self.detect_xmin = self.config['dice_cup_base_detection_area']['xmin']
        self.detect_ymin = self.config['dice_cup_base_detection_area']['ymin']
        self.detect_xmax = self.config['dice_cup_base_detection_area']['xmax']
        self.detect_ymax = self.config['dice_cup_base_detection_area']['ymax']
        self.load_model()

    def load_model(self):
        model_path  = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', 'yolo', 'dice_cup_base.pt')
        self.model = YOLO(model_path).to('cpu')

    def detect(self, frame):
        detect_area = frame[self.detect_ymin:self.detect_ymax, self.detect_xmin:self.detect_xmax]
        results = self.model(detect_area, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()  # (xmin, ymin, xmax, ymax, confidence, class)
        for base in detections:
            xmin, ymin, xmax, ymax, cnf, cls = base
            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

            box_width = xmax - xmin

            if box_width > 450:
                return True, detect_area, xmin, ymin, xmax, ymax
            
        return False, None, 0, 0, 0, 0

