FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 \
    nvidia-utils-535 \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["python3", "app.py"]
