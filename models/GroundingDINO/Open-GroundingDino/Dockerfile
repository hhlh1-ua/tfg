FROM huggingface/transformers-pytorch-gpu:4.41.3 

RUN apt-get update 
RUN apt-get install -y git gcc libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg
RUN python3 -m pip install --upgrade pip


ENV CUDA_HOME=/usr/local/cuda
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
