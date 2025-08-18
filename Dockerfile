FROM python:3.12-slim

# 安裝系統依賴（build-essential 等等）
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libhdf5-dev \
    libglib2.0-0 \
    pkg-config \  
    && rm -rf /var/lib/apt/lists/*

# 建立工作目錄
WORKDIR /app

# 複製所有檔案到容器
COPY . /app

# 安裝 Python 套件
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]