import pymysql
import logging
import os

class Database:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config['database']

    def connect_db(self):
        """建立並回傳一個 MySQL 連線"""
        try:
            return pymysql.connect(
                host=self.config["host"],
                user=self.config["credentials"]["username"],
                password=self.config["credentials"]["password"],
                database=self.config["dbname"],
                port=self.config.get("port", 3306),
                cursorclass=pymysql.cursors.DictCursor
            )
        except Exception as e:
            self.logger.error(f"[DB] Failed to connect to DB: {e}")
            return None
        
    def insert_log(self, dices, system_time):
        """
        插入一筆記錄：
        - `DICES`: 需要為字串
        - `DETECTEDTIME`: 可能為 None
        - `SYSTEMTIME`: 轉換為 UTC+8 的時間字串
        """
        conn = self.connect_db()
        if not conn:
            return
        
        try:
            with conn.cursor() as cursor:
                sql = f"""
                INSERT INTO `{self.config['table']}` (dicelist, systemtime)
                VALUES (%s, %s)
                """
                cursor.execute(sql, (dices, system_time))
                conn.commit()
                self.logger.info("[DB] Insert OK!")
        except Exception as e:
            self.logger.error(f"[DB] Insert Failed: {e}")
        finally:
            conn.close()  # 確保連線被關閉