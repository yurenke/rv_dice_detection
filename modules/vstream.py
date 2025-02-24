import threading
import time
import cv2

class VideoStream:
    def __init__(self, logger, config):
        """
        :param src: 影片來源 (0 代表攝影機，或串流 URL)
        :param reconnect_interval: 重連間隔 (秒)
        :param max_fail_count: 允許 cap.read() 失敗的次數，超過則重連
        """
        self.logger = logger
        vs_config = config['vstream']
        self.src = vs_config['url']
        self.max_fail_count = vs_config['max_fail_count']
        self.reconnect_interval = vs_config['reconnect_interval']
        self.fail_count = 0
        self.stopped = False
        self.connected = threading.Event()
        self.reset_needed = threading.Event()
        self.lock = threading.Lock()
        self.cap = None
        self.ret = False
        self.frame = None

        # 啟動影像讀取執行緒
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def connect(self):
        with self.lock:
            self.ret = False
            self.frame = None  # 清空影格，避免讀到過時畫面
        """嘗試建立串流連線"""
        if self.cap:
            self.cap.release()  # 釋放舊資源

        while True:
            self.logger.info(f"[VS] Trying to connect to {self.src}")
            self.cap = cv2.VideoCapture(self.src)
            if self.cap.isOpened():
                self.logger.info("[VS] Stream connected !!")
                
                for _ in range(5):  # 最多嘗試 5 次
                    ret, _ = self.cap.read()
                    if ret:
                        self.logger.info("[VS] Successfully read the 1st frame..ready to read")
                        break
                    time.sleep(0.5)  # 等待 0.5 秒再試

                self.fail_count = 0  # 重置失敗計數
                self.connected.set()  # 影像讀取穩定後才通知主程序
                return

            self.logger.error(f"[VS] Failed to connect...Retry after {self.reconnect_interval} secs")
            time.sleep(self.reconnect_interval)  # 等待config的重連時間後重試         

    def update(self):
        """背景執行緒持續讀取最新影格"""
        self.connect()  # 先嘗試連線

        while not self.stopped:
            try:
                if not self.cap.isOpened():
                    self.logger.error("[VS] cap.isOpened() returns False. Try to reconnect")
                    self.connected.clear()  # 讓主程序等待connected再次被set()
                    self.reset_needed.set()
                    self.connect()
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    self.fail_count += 1
                    self.logger.warning(f"[VS] cap.read() failed {self.fail_count}/{self.max_fail_count}")
                    
                    if self.fail_count >= self.max_fail_count:
                        self.logger.error("[VS] cap.read() failed too many times... Try to reconnect")
                        self.connected.clear()
                        self.reset_needed.set()
                        self.connect()                
                    continue  # 跳過這一幀，嘗試讀取下一幀

                # 讀取成功，重置 fail_count
                self.fail_count = 0

                # 鎖定寫入最新影格
                with self.lock:
                    self.ret, self.frame = ret, frame

            except Exception as e:
                self.logger.error(f"[VS] thread update Exception occurs: {e}")
                self.connected.clear()
                self.reset_needed.set()
                time.sleep(5)
                self.connect()           

    def read(self):
        """取得最新影格"""
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        """停止串流"""
        self.stopped = True
        self.thread.join()
        if self.cap:
            self.cap.release()
