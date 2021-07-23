#!/bin/bash

# Check driver & Install CUDA
if ! command -v nvidia-smi &> /dev/null
then
    echo "You must first install an NVIDIA driver"
fi

# Install python
if ! python3 --version 2>&1 | grep '3.6';
then
    echo "Installing Python 3.6"
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update --fix-missing
    sudo apt-get install -y python3.6 python3.6-distutils python3.6-dev
    sudo python3.6 -m easy_install pip
else
    echo "Python installed"
fi

# Install ffmpeg & soundfile
sudo apt-get install -y ffmpeg libsndfile1

# Install Dependencies
python3.6 -m pip install -r requirements.txt
echo "Successfully installed python packages"
echo "Install complete"
