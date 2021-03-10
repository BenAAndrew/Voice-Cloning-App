#!/bin/bash

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

# Install Pytorch
PYTORCH=`pip list | grep torch`
if [[ -z "$PYTORCH" ]]
then
    echo "Installing Pytorch"
    pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
else
    echo "Pytorch installed"
fi

# Install ffmpeg
sudo apt-get install -y ffmpeg

# Install Dependencies
pip install -r requirements.txt
echo "Successfully installed python packages"
echo "Install complete"

fi