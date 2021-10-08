FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Lib dependencies
RUN apt-get update
RUN apt-get install -y ffmpeg build-essential

# Setup
WORKDIR /app
COPY application/ /app/application
COPY dataset/ /app/dataset
COPY training/ /app/training
COPY synthesis/ /app/synthesis
COPY main.py /app
COPY requirements.txt /app

# Python dependencies
RUN pip install -r requirements.txt

# Start app
CMD [ "python3", "main.py" ]
