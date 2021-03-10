#!/bin/bash

# git submodule update --init

# Check driver & Install CUDA
if ! command -v nvidia-smi &> /dev/null
then
    echo "You must first install an NVIDIA driver"
else
    if command -v nvcc &> /dev/null
    then
        echo "CUDA installed"   
    else
        echo "Installing CUDA"
        sudo add-apt-repository -y ppa:graphics-drivers
        sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
        sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
        sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

        sudo apt install -y nvidia-cuda-toolkit
        sudo apt-get install -y libcudnn7

    if command -v nvcc &> /dev/null
    then
        echo "Successfully installed CUDA, please restart PC"
    else
        echo "Failed to install CUDA"
        exit 1
    fi
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

# Install Pytorch
PYTORCH=`pip list | grep torch`
if [[ -z "$PYTORCH" ]]
then
    echo "Installing Pytorch"
    CUDA=`nvcc --version`
    if [[ $CUDA == *"10.1"* ]]
    then
        pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    elif [[ $CUDA == *"10.2"* ]]
    then
        pip install torch==1.7.1 torchvision==0.8.2
    elif [[ $CUDA == *"11.0"* ]]
    then
        pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    else
        echo "Invalid version of cuda (not 10.1-11.0)"
        exit 1
    fi
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