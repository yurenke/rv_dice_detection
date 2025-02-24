CREATE DATABASE IF NOT EXISTS dicedb;
USE dicedb;
CREATE TABLE dice_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dicelist CHAR(17) NOT NULL,
    detectedtime DATETIME NULL,
    systemtime DATETIME NOT NULL
);

CREATE INDEX idx_systemtime ON dice_results(systemtime);

-- 創建新的用戶 'myuser' 並設定密碼 'mypassword'
CREATE USER 'diceuser'@'%' IDENTIFIED BY '12345';

-- 允許 myuser 存取 `mydatabase` 資料庫
GRANT ALL PRIVILEGES ON dicedb.* TO 'diceuser'@'%';

-- 讓權限變更生效
FLUSH PRIVILEGES;