#!/bin/bash

# Check driver & Install CUDA
if ! command -v nvidia-smi &> /dev/null
then
    echo "You must first install an NVIDIA driver"
fi

# Install python
if ! command -v python3 &> /dev/null
then
    echo "Installing Python 3.8"
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.8
else
    echo "Python installed"
fi

# Install ffmpeg
sudo apt-get install -y ffmpeg

# Install Dependencies
pip install -r requirements.txt
echo "Successfully installed python packages"
echo "Install complete"

fi