FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# Setup
RUN apt-get update
WORKDIR /app
COPY . /app

# FFMPEG
RUN apt-get install -y ffmpeg

# Python dependencies
RUN pip install -r requirements.txt

# Start app
ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]
