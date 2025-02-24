import cv2
import os
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
from datetime import datetime

from modules.ocr import OCRObserver
from modules.vstream import VideoStream
from modules.detector import DiceDetector
from modules.db import Database

# 定義骰子狀態
class DiceState(IntEnum):
    STILL = 0    # 靜止
    ROLLING = 1  # 滾動中

DICE_VALUE_STR = {0.0: '1', 1.0: '2', 2.0: '3', 3.0: '4', 4.0: '5', 5.0: '6'}

class DiceApp:
    def __init__(self):
        config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'configs', 'config.yaml')
        self.config = self.load_config(config_path)
        self.setup_logging()

        self.dice_state = DiceState.STILL
        self.stable_count = 0
        self.stable_threshold = self.config['detector']['stable_threshold']
        self.previous_positions = None
        self.row_height = (self.config['detector']['detect_area']['ymax'] - self.config['detector']['detect_area']['ymin']) / self.config['detector']['rows']
        self.col_width = (self.config['detector']['detect_area']['xmax'] - self.config['detector']['detect_area']['xmin']) / self.config['detector']['cols']
        
        self.db = Database(self.logger, self.config)
        self.ocr = OCRObserver(self.logger)
        self.detector = DiceDetector(self.config)
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
        self.previous_positions = None

    def get_row_column(self, x, y):
        # 計算x和y所在的row和column
        row = int(y // self.row_height)  # 以y座標計算行數
        col = int(x // self.col_width)  # 以x座標計算列數

        # 確保row和col在範圍內
        row = max(0, min(row, self.config['detector']['rows'] - 1))
        col = max(0, min(col, self.config['detector']['cols'] - 1))

        return row, col
    
    def write_result(self, sorted_index, positions, detections, frame):
        # 骰子值和位在哪一個row和col
        value_row_col = []
        for i in range(len(positions)):
            v = DICE_VALUE_STR[detections[sorted_index[i]][5]]
            r, c = self.get_row_column(positions[i][0], positions[i][1])
            value_row_col += [f"{v},{r},{c}"]

        dices_str = ';'.join(value_row_col)
        # 偵測畫面時間
        detected_time_str = self.ocr.get_datetime(frame)

        # 取得當前系統時區的時間
        local_tz = get_localzone()  # 取得系統時區
        system_time = datetime.now(local_tz)  # 設定為系統時區的 datetime
        system_time_utc8 = system_time.astimezone(ZoneInfo("Asia/Shanghai"))  # 轉為 UTC+8
        system_time_str = system_time_utc8.strftime("%Y-%m-%d %H:%M:%S")  # 轉為字串

        self.logger.info(f"[DiceApp] dices: {dices_str} detected_time: {detected_time_str}")

        # 窵入DB
        self.db.insert_log(dices_str, system_time_str, detected_time_str)

    def detect_dice(self, frame):
        # 偵測骰子
        detections = self.detector.detect(frame)
        num_dice = len(detections)

        # 判斷骰子狀態
        if num_dice != 3:
            self.stable_count = 0  # 重新計算穩定次數
            if self.dice_state == DiceState.STILL:
                self.logger.info("[DiceApp] Dice rolling detected!!")
                self.dice_state = DiceState.ROLLING
        else:
            bottom_centers = []
            for dice in detections:
                xmin, ymin, xmax, ymax, confidence, cls = dice

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2

                bottom_centers.append((x_center, y_center))    
            current_positions = np.array(bottom_centers)
            sorted_index = np.lexsort(current_positions.T) # 先根據 x 排序，若 x 相同則用 y 排序
            current_positions = current_positions[sorted_index]

            if self.previous_positions is not None and not np.allclose(self.previous_positions, current_positions, atol=5):
                self.stable_count = 0  # 重新計算穩定次數
                if self.dice_state == DiceState.STILL:
                    self.logger.info("[DiceApp] Dice rolling detected!!")
                    self.dice_state = DiceState.ROLLING
            else:
                self.stable_count = max(self.stable_count + 1, self.stable_threshold)
                # print(f"dice remain still {stable_count}/{STABILITY_THRESHOLD}")
                if self.stable_count >= self.stable_threshold:
                    if self.dice_state == DiceState.ROLLING:
                        self.logger.info(f"[DiceApp] Dice roll result confirmed!! pos: {current_positions}")
                        self.dice_state = DiceState.STILL
                        self.write_result(sorted_index, current_positions, detections, frame)
                else:
                    if self.dice_state == DiceState.STILL:
                        self.dice_state = DiceState.ROLLING

            # 更新前一次的位置
            self.previous_positions = current_positions

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
