# Install
After installing & running the app should open in your browser at http://localhost:5000/

### Windows
Download and Run the latest executable from [Releases](https://github.com/BenAAndrew/Voice-Cloning-App/releases)

### Linux
1. Clone this repository
2. Run `./install.sh` from the root of the repository
3. Run `python3.6 main.py`

### Manual Install (Linux/ Windows)
1. Clone this repository
2. Install [Python](https://www.python.org/) (version 3.6)
3. Windows only: Install Visual Studio 2019 with the following components:
    - MSVC toolset C++ 2019 v142 (x86,x64) latest
    - Visual C++ 2019 Redistributable Update
    - Windows 10 SDK (10.0.17763.0)
4. Run `pip install -r requirements.txt`
5. Run `python main.py`

### Docker
1. Clone this repository
2. Run `docker build -t voice-cloning:latest .`
3. Run `docker run -d -p 5000:5000 voice-cloning`

## Install CPU Only version
**Please Note:** The CPU Only version supports all features except local training

### Windows
Download and Run the latest `cpuonly` executable from [Releases](https://github.com/BenAAndrew/Voice-Cloning-App/releases)

### Linux
1. Clone this repository
2. Install [Python](https://www.python.org/) (version 3.6)
3. Run `pip install -r requirements-cpu.txt`
4. Run `python main.py`
