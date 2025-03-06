from typing import Any
import os
import torch
import cv2
from ultralytics import YOLO
from PIL import Image
from parseq.strhub.data.module import SceneTextDataModule
from datetime import datetime
from zoneinfo import ZoneInfo
import re

class OCRObserver():
    parseq = None
    parseq_img_transform = None

    def __init__(self, logger) -> None:
        self.logger = logger
        self.load_yolo()
        self.load_parseq()

    def load_yolo(self):
        path_yolov8  = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', 'yolo', 'search_panel.pt')
        self.model_yolo_panel = YOLO(path_yolov8).to('cpu')
        # path_yolov8  = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', 'yolo', 'get_datetime.pt')
        # self.model_yolo_datetime = YOLO(path_yolov8).to('cpu')

    def load_parseq(self):
        _local_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'parseq')
        _pt_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', 'parseq.pt')
        self.parseq = torch.hub.load(_local_path, 'parseq', pretrained=True, source='local', weight_file=_pt_file_path).eval()  
        self.parseq_img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)

    def get_position_by_yolo_results(self, results:list, names:list, target_name:str, confident_threshold:float = 0.5) -> list:
        for res in results:
            for box in res.boxes:
                xyxy = [int(_) for _ in box.xyxy[0]]
                cint = int(box.cls)
                cname = names[cint]
                confident = float(box.conf)
                if (cname == target_name) and confident > confident_threshold:
                    return xyxy
        return [0,0,0,0]
    
    def get_cropped_image_by_position(self, image:any, xyxy:list):
        if xyxy[0] == xyxy[2]:
            return None
        return image[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]

    def crop_panel_from_frame(self, frame) ->  tuple[Any, list]:
        """ use yolo model search a panel in frame

        return (
            croped: a numpy data which is panel liked in frame,
            xyxy: position shape of panel,
        )
        """
        results = self.model_yolo_panel(frame, verbose=False)
        xyxy = self.get_position_by_yolo_results(results, names=self.model_yolo_panel.names, target_name='panel')
        image = self.get_cropped_image_by_position(image=frame, xyxy=xyxy)
        
        return image, xyxy
    
    def get_datetime_from_panel(self, panel) -> tuple[Any, list]:
        """ use yolo model get datetime in panel images
        
        """
        h, w, _ = panel.shape

        return panel[int(h / 3): int(h * 2 / 3), :]
        # results = self.model_yolo_datetime(panel, verbose=False)
        # xyxy = self.get_position_by_yolo_results(results, names=self.model_yolo_datetime.names, target_name='datetime')
        # image = self.get_cropped_image_by_position(image=panel, xyxy=xyxy)
        
        # return image, xyxy

    def parseq_parse(self, img):
        
        img = Image.fromarray(img).convert('RGB')
        img = self.parseq_img_transform(img).unsqueeze(0)

        logits = self.parseq(img)

        pred = logits.softmax(-1)
        labels, confidence = self.parseq.tokenizer.decode(pred)

        return labels
    
    def get_datetime(self, img):
        dt_str = None
        panel, _ = self.crop_panel_from_frame(img)

        if panel is None:
            self.logger.warning('[OCR] panel not found')

        if panel is not None:
            img_dt = self.get_datetime_from_panel(panel)

            if img_dt is not None:
                labels = self.parseq_parse(img_dt)
                if not labels or not labels[0].startswith('CCT'):
                    return None
                date_str = labels[0][3:]
                date_str = re.sub(r"[^0-9/]", ":", date_str) # 避免可能的辨識錯誤，例如:被辨識成;或.
                try:
                    dt = datetime.strptime(date_str, "%y/%m/%d%H:%M")
                except ValueError:
                    self.logger.error(f"Failed to parse detected datetime string: {date_str}")
                    return None

                # 設定為上海時區，因平板顯示為CCT時間
                dt_cct = dt.replace(tzinfo=ZoneInfo("Asia/Shanghai"))

                # 風控需求，轉為美東時間
                dt_est = dt_cct.astimezone(ZoneInfo("America/New_York"))

                # 取出美東時間表示字串去除時區資訊
                dt_str = dt_est.strftime("%Y-%m-%d %H:%M:%S")

        return dt_str