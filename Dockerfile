#使用 Python 3.9 映像檔作為基底
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製本地的程式到容器內
COPY . .

# 安裝必要的 Python 套件
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 開放容器內部的 5000 端口
# EXPOSE 5000

# 啟動程式
CMD ["python", "app.py"]
