#!/bin/bash

# Check driver & Install CUDA
if ! command -v nvidia-smi &> /dev/null
then
    echo "You must first install an NVIDIA driver"
fi

# Install python
if ! python3 --version 2>&1 | grep '3.8';
then
    echo "Installing Python 3.8"
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update --fix-missing
    sudo apt-get install -y python3.8 python3.8-distutils python3.8-dev
    sudo python3.8 -m easy_install pip
else
    echo "Python installed"
fi

# Install ffmpeg
sudo apt-get install -y ffmpeg

# Install Dependencies
python3.8 -m pip install -r requirements.txt
echo "Successfully installed python packages"
echo "Install complete"
