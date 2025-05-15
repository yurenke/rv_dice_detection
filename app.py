import cv2
import os
import shutil
from itertools import permutations
import threading
import time
import torch
import numpy as np
from enum import IntEnum
import yaml
import logging
from logging.handlers import TimedRotatingFileHandler
from ultralytics import YOLO
from tzlocal import get_localzone
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from glob import glob

from modules.ocr import OCRObserver
from modules.vstream import VideoStream
from modules.detector import DiceDetector, DiceCupBaseDetector
from modules.db import Database

# 定義骰子狀態
class DiceState(IntEnum):
    STILL = 0    # 靜止
    ROLLING = 1  # 滾動中

DICE_VALUE_MAP = {0.0: 1, 1.0: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}

class DiceApp:
    def __init__(self):
        config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'configs', 'config.yaml')
        self.config = self.load_config(config_path)
        self.setup_logging()

        self.dice_state = DiceState.STILL
        self.stable_count = 0
        self.stable_threshold = self.config['detector']['stable_threshold']
        self.prev_dice = None
        # self.prev_report_time = 0
        # self.previous_report = None
        self.detect_xmin = self.config['detector']['dice_cup_base_detection_area']['xmin']
        self.detect_ymin = self.config['detector']['dice_cup_base_detection_area']['ymin']
        self.detect_xmax = self.config['detector']['dice_cup_base_detection_area']['xmax']
        self.detect_ymax = self.config['detector']['dice_cup_base_detection_area']['ymax']
        self.pixel_diff_threshold = self.config['detector']['pixel_diff_threshold']

        self.image_dir = self.config['images']['save_dir']
        self.image_keep_days = self.config['images']['keep_days']
        self.traindata_dir = self.config['traindata']['save_dir']
        
        self.db = Database(self.logger, self.config)
        # self.ocr = OCRObserver(self.logger)
        self.dice_cup_base_detector = DiceCupBaseDetector(self.config)
        self.dice_detector = DiceDetector()
        self.logger.info("[DiceApp] models loaded !! Starting Video Stream Connection...")
        
        self.vs = VideoStream(self.logger, self.config)
    
    def load_config(self, config_path):
        """讀取 YAML 設定檔"""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def setup_logging(self):
        """設定 logging"""
        log_dir = self.config["logging"]["log_dir"]
        log_level = self.config["logging"]["level"]
        backup_days = self.config["logging"]["backup_days"]

        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "app.log")

        log_formatter = logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s: %(message)s")

        # 設定 TimedRotatingFileHandler（每天切割，最多保留指定天數）
        file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=backup_days, encoding="utf-8")
        file_handler.setFormatter(log_formatter)

        # 設定 root logger
        logging.basicConfig(level=getattr(logging, log_level), handlers=[file_handler])
        self.logger = logging.getLogger("app")

    def reset_state_variables(self):
        self.state = DiceState.STILL
        self.stable_count = 0
        self.prev_dice = None
        # self.prev_report_time = 0
        # self.previous_report = None

    def get_row_column(self, x, y, row_height, col_width):
        # 計算x和y所在的row和column
        row = int(y // row_height)  # 以y座標計算行數
        col = int(x // col_width)  # 以x座標計算列數

        # 確保row和col在範圍內
        row = max(0, min(row, self.config['detector']['rows'] - 1))
        col = max(0, min(col, self.config['detector']['cols'] - 1))

        return row, col
    
    def save_image_by_date(self, image, image_with_boxes, detections, time_str, date_str, postfix_str):
        # 取得今天日期字串
        # today_str = datetime.now().strftime("%Y-%m-%d")
        img_with_boxes_save_dir = os.path.join(self.image_dir, date_str)
        os.makedirs(img_with_boxes_save_dir, exist_ok=True)
        traindata_date_dir = os.path.join(self.traindata_dir, date_str)
        os.makedirs(traindata_date_dir, exist_ok=True)
        traindata_img_dir = os.path.join(traindata_date_dir, 'images')
        os.makedirs(traindata_img_dir, exist_ok=True)
        traindata_label_dir = os.path.join(traindata_date_dir, 'labels')
        os.makedirs(traindata_label_dir, exist_ok=True)

        # 存image with bounding boxes, dice bottom centers, and dice cup base boxes
        filename = f"{time_str}_{postfix_str}.jpg"
        filepath = os.path.join(img_with_boxes_save_dir, filename)
        cv2.imwrite(filepath, image_with_boxes)

        # 存original image
        filename = f"{time_str}.jpg"
        filepath = os.path.join(traindata_img_dir, filename)
        cv2.imwrite(filepath, image)

        # 存labels
        filename = f"{time_str}.txt"
        filepath = os.path.join(traindata_label_dir, filename)

        img_h, img_w = image.shape[:2]
        with open(filepath, "w") as f:
            for det in detections:
                xmin, ymin, xmax, ymax, conf, cls = det
                class_id = int(cls)
                # 轉成 yolo 格式：中心點座標、寬高
                x_center = (xmin + xmax) / 2 / img_w
                y_center = (ymin + ymax) / 2 / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # 清除過期資料夾
        self.cleanup_old_folders(self.image_dir, self.traindata_dir, date_str, self.image_keep_days)

    def cleanup_old_folders(self, img_dir, traindata_dir, current_date_str, keep_days):
        today = datetime.strptime(current_date_str, "%Y-%m-%d").date()
        img_folders = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
        traindata_folders = [f for f in os.listdir(traindata_dir) if os.path.isdir(os.path.join(traindata_dir, f))]

        for folder in img_folders:
            try:
                folder_date = datetime.strptime(folder, "%Y-%m-%d").date()
                if (today - folder_date).days > keep_days:
                    full_path = os.path.join(img_dir, folder)
                    print(f"Deleting old folder: {full_path}")
                    shutil.rmtree(full_path)
                    # for f in glob(os.path.join(full_path, "*")):
                    #     os.remove(f)
                    # os.rmdir(full_path)
            except ValueError:
                continue  # 如果資料夾不是日期格式，就跳過

        for folder in traindata_folders:
            try:
                folder_date = datetime.strptime(folder, "%Y-%m-%d").date()
                if (today - folder_date).days > keep_days:
                    full_path = os.path.join(traindata_dir, folder)
                    print(f"Deleting old folder: {full_path}")
                    shutil.rmtree(full_path)
                    # for f in glob(os.path.join(full_path, "*")):
                    #     os.remove(f)
                    # os.rmdir(full_path)
            except ValueError:
                continue  # 如果資料夾不是日期格式，就跳過
    
    def write_result(self, curr_dice, detections, frame, dice_zone, base_xmin, base_ymin, base_xmax, base_ymax):
        # 骰子值和位在哪一個row和col
        base_xmin, base_ymin, base_xmax, base_ymax = map(int, [base_xmin, base_ymin, base_xmax, base_ymax])
        h, w = base_ymax - base_ymin, base_xmax - base_xmin
        row_h, col_w = h // self.config['detector']['rows'], w // self.config['detector']['cols']

        value_row_col = []
        for i in range(len(curr_dice)):
            v = DICE_VALUE_MAP[curr_dice[i][2]]
            r, c = self.get_row_column(max(0, curr_dice[i][0] - base_xmin), max(0, curr_dice[i][1] - base_ymin), row_h, col_w)
            value_row_col += [(v, r, c)]

        sorted_value_row_col = sorted(value_row_col, key= lambda x: (x[1], x[2], x[0]))
        str_value_row_col, postfix_value_row_col = [], []
        for v, r, c in sorted_value_row_col:
            str_value_row_col += [f"{v},{r},{c}"]
            postfix_value_row_col += [f"{v}{r}{c}"]

        dices_str = ';'.join(str_value_row_col)
        postfix_str = '_'.join(postfix_value_row_col)
        # 偵測畫面時間
        # detected_time_str = self.ocr.get_datetime(frame)

        # 取得當前系統時區的時間
        local_tz = get_localzone()  # 取得系統時區
        system_time = datetime.now(local_tz)  # 設定為系統時區的 datetime
        system_time_utc8 = system_time.astimezone(ZoneInfo("Asia/Shanghai"))  # 轉為 UTC+8
        system_time_str = system_time_utc8.strftime("%Y-%m-%d %H:%M:%S")  # 轉為字串

        # system_time_utc_minus_4 = system_time.astimezone(ZoneInfo("America/New_York"))  # 轉為 UTC-4/-5
        # detected_time_str = system_time_utc_minus_4.strftime("%Y-%m-%d %H:%M:%S")  # 轉為字串

        system_time_for_image_str = system_time_utc8.strftime("%Y-%m-%d-%H-%M-%S")
        system_date_for_image_str = system_time_utc8.strftime("%Y-%m-%d")

        self.logger.info(f"[DiceApp] dices: {dices_str} system_time: {system_time_str}")

        # 窵入DB
        self.db.insert_log(dices_str, system_time_str)

        dice_zone_with_boxes = dice_zone.copy()

        # 繪製格線，骰子外框，及底部中心點，再保存圖片
        cv2.rectangle(dice_zone_with_boxes, (base_xmin, base_ymin), (base_xmax, base_ymax), (0, 255, 0), 2)

        # 畫水平分割線
        for i in range(1, self.config['detector']['rows']):
            y = i * row_h + base_ymin
            cv2.line(dice_zone_with_boxes, (base_xmin, y), (base_xmax, y), (0, 255, 0), 2)  # 綠色線

        # 畫垂直分割線
        for j in range(1, self.config['detector']['cols']):
            x = j * col_w + base_xmin
            cv2.line(dice_zone_with_boxes, (x, base_ymin), (x, base_ymax), (0, 255, 0), 2)  # 綠色線

        for dice in detections:
            xmin, ymin, xmax, ymax, confidence, cls = dice
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            # 繪製骰子邊界框
            cv2.rectangle(dice_zone_with_boxes, (xmin, ymin), (xmax, ymax), (255, 200, 100), 2)

        for x, y, _ in curr_dice:
            x, y = int(x), int(y)
            cv2.circle(dice_zone_with_boxes, center=(x, y), radius=5, color=(255, 0, 0), thickness=-1)

        self.save_image_by_date(dice_zone, dice_zone_with_boxes, detections, system_time_for_image_str, system_date_for_image_str, postfix_str)

    # def is_duplicate(self, current_pos, dice_points):
    #     if not self.previous_report:
    #         return False
        
    #     if np.array_equal(self.previous_report['points'], dice_points) and np.allclose(self.previous_report['positions'], current_pos, atol=self.pixel_diff_threshold):
    #         self.logger.warning("[DiceApp] duplicate result, don't report !!")
    #         return True
        
    #     return False
    
    def is_same_dice(self, curr_dice):
        for perm in permutations(curr_dice):
            match = True
            for i in range(3):
                prev = self.prev_dice[i]
                curr = perm[i]
                pos_diff = np.linalg.norm(prev[:2] - curr[:2])
                if curr[2] != prev[2] or pos_diff > self.pixel_diff_threshold:
                    match = False
                    break
            if match:
                return True
        return False


    def detect_dice(self, frame):
        found_dice_cup_base, dice_detect_zone, base_xmin, base_ymin, base_xmax, base_ymax = self.dice_cup_base_detector.detect(frame)
 
        if not found_dice_cup_base:
            self.logger.warning(f"[DiceApp] Dice cup base not found !!")
            return
        # 偵測骰子
        detections = self.dice_detector.detect(dice_detect_zone)
        num_dice = len(detections)

        # 判斷骰子狀態
        if num_dice != 3:
            self.stable_count = 0  # 重新計算穩定次數
            self.prev_dice = None
            if self.dice_state == DiceState.STILL:
                self.logger.info(f"[DiceApp] Dice rolling detected!! num_dice != 3 num_dice: {num_dice}")
                self.dice_state = DiceState.ROLLING
        else:
            # bottom_centers, abs_bottom_centers = [], []
            curr_dice = []
            for dice in detections:
                xmin, ymin, xmax, ymax, confidence, cls = dice

                x_center = (xmin + xmax) / 2
                # y_center = (ymin + ymax) / 2
                height = ymax - ymin
                y_center = ymax - height * (1/5)

                curr_dice.append([x_center, y_center, cls])

            curr_dice = np.array(curr_dice)
            # current_abs_positions = current_abs_positions[sorted_index]

            if self.prev_dice is not None and not self.is_same_dice(curr_dice):
                self.stable_count = 0  # 重新計算穩定次數
                if self.dice_state == DiceState.STILL:
                    self.logger.info("[DiceApp] Dice rolling detected!! case 2")
                    self.logger.info(f"prev_dice: {self.prev_dice}")
                    self.logger.info(f"curr_dice: {curr_dice}")
                    self.dice_state = DiceState.ROLLING
            else:
                self.stable_count = max(self.stable_count + 1, self.stable_threshold)
                # print(f"dice remain still {stable_count}/{STABILITY_THRESHOLD}")
                if self.stable_count >= self.stable_threshold:
                    if self.dice_state == DiceState.ROLLING:
                        self.logger.info(f"[DiceApp] Dice roll result confirmed!! curr_dice: {curr_dice}")
                        self.dice_state = DiceState.STILL

                        # curr_time = time.time()  
                        # if curr_time - self.prev_report_time > 30:# if not self.is_duplicate(current_positions, dice_points):
                        self.write_result(curr_dice, detections, frame, dice_detect_zone, base_xmin, base_ymin, base_xmax, base_ymax)
                            # self.prev_report_time = curr_time
                        # else:
                        #     self.logger.warning("[DiceApp] duplicate, don't report")
                            # self.previous_report = {'points': dice_points, 'positions': current_positions}
                else:
                    if self.dice_state == DiceState.STILL:
                        self.logger.info(f"[DiceApp] stable count < threshold!! stable_count: {self.stable_count}")
                        self.stable_count = 0
                        self.dice_state = DiceState.ROLLING

            # 更新前一次的位置
            self.prev_dice = curr_dice

    def run(self):
        last_detection_time = 0

        while True:
            try: 
                self.vs.connected.wait()  # 等待相機連線成功或重新連線成功
                if self.vs.reset_needed.is_set():
                    self.logger.warning("[DiceApp] Reconnection occurred! Reset states")
                    self.reset_state_variables()
                    self.vs.reset_needed.clear()

                ret, frame = self.vs.read()
                if not ret:
                    self.logger.warning("[DiceApp] cannot get frame...")
                    time.sleep(2)
                    continue  # 跳過此次迴圈

                frame = frame.copy()

                current_time = time.time()
                if current_time - last_detection_time >= self.config['detector']['check_interval']:
                    last_detection_time = current_time
                    # detecting dices
                    self.detect_dice(frame)
            except Exception as e:
                self.logger.error(f'[DiceApp] Exception occurs: {e}')
                time.sleep(2)

            # cv2.imshow("Stream", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # vs.stop()
        # cv2.destroyAllWindows()

# === 主程式 ===
if __name__ == "__main__":
    app = DiceApp()
    app.run()
