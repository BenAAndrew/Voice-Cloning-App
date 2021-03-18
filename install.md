# Install
**This project requires an NVIDIA GPU for training which can support version 440.33+ and at least 4GB GPU memory**

After installing & running the app should open in your browser at http://localhost:5000/

### Windows
1. Clone this repository
2. Right click `install.ps1` and select "Run with powershell"
3. Click next through the CUDA installer (leave default values)
4. Download and Run latest executable from [Releases](https://github.com/BenAAndrew/Voice-Cloning-App/releases)

### Linux
1. Clone this repository
2. Run `./install.sh` from the root of the repository
3. Run `python main.py`

### Docker
1. Clone this repository
2. Run `docker build -t voice-cloning:latest .`
3. Run `docker run -d -p 5000:5000 voice-cloning`

### Manual Install (Linux/ Windows)
1. Clone this repository
2. Install [CUDA](https://developer.nvidia.com/cuda-zone) (version 10.2+)
3. Install [Python](https://www.python.org/) (version 3.4-3.8)
4. Install [Pytorch](https://pytorch.org/) (ensure cuda version matches your install)
5. Run `pip install -r requirements.txt`
5. Run `python main.py`

## Install Demo

**You only need to install the demo if you haven't done the main install. This is kept as a seperate install for CPU only machines which demo completed voices.**

**You cannot train with this install (only for synthesis)**

### Linux
1. Clone this repository
2. Run `./install_cpu_only.sh` from the root of the repository
3. Run `python main.py`

### Windows
1. Clone this repository
2. Run `install_cpu_only.ps1` by right clicking and selecting "Run with Powershell"
3. Run `python main.py`
