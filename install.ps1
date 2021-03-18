function Error {
    param (
        $message
    )
    Write-Output $message
    Start-Sleep -s 5
    throw $message
}

function Check-Installed {
    param (
        $app
    )
    $output = cmd.exe /c $app /w 2>&1
    return -Not ($output -like "*is not recognized*" -or $output -like "*was not found*")
}

<# CUDA install #>
if (-Not (Check-Installed "nvcc")){
    Write-Host "Downloading CUDA 11.0..."
    Invoke-WebRequest -Uri "https://developer.download.nvidia.com/compute/cuda/11.0.2/network_installers/cuda_11.0.2_win10_network.exe" -OutFile "cuda-11.exe"
    Write-Host "Installing CUDA 11.0"
    Start-Process -Wait -FilePath "cuda-11.exe"
    Remove-Item -Path cuda-11.exe
    Write-Host "Installed CUDA 11.0"
} else {
    Write-Host "CUDA already installed"
}

<# Python install #>
# if (-Not (Check-Installed "python")){
#     Write-Host "Downloading Python 3.8..."
#     Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.8.7/python-3.8.7-amd64.exe" -OutFile "python-3.8.7.exe"
#     Write-Host "Installing Python 3.8"
#     $pyInstallOutput = cmd.exe /c python-3.8.7.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
#     Remove-Item -Path python-3.8.7.exe
#     Write-Host "Installed Python 3.8"
# } else {
#     Write-Host "Python already installed"
# }

# <# Pytorch install #>
# $nvccOutput = cmd.exe /c nvcc --version /w 2>&1
# Write-Host "Installing Pytorch..."
# if ($nvccOutput -like "*10.1*"){
#     $pytorchOutput = cmd.exe /c pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# } elseif ($nvccOutput -like "*10.2*"){
#     $pytorchOutput = cmd.exe /c pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# } elseif ($nvccOutput -like "*11.0*"){
#     $pytorchOutput = cmd.exe /c pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# } else {
#     Error "Unsupported CUDA version (not 10.1-11.0)"
# }
# Write-Host "Pytorch installed"

<# FFMPEG install #>
if (-Not (Check-Installed "ffmpeg")){
    Write-Host "Downloading FFMPEG..."
    Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile "ffmpeg.zip"
    Write-Host "Installing FFMPEG..."
    Expand-Archive -Path ffmpeg.zip -DestinationPath C:\
    Rename-Item C:\ffmpeg-4.3.2-2021-02-27-essentials_build C:\ffmpeg
    [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ffmpeg\bin", "User")
    Remove-Item -Path ffmpeg.zip
    Write-Host "Installed FFMPEG" 
} else {
    Write-Host "FFMPEG installed"
}

# <# Dependency install #>
# Write-Host "Installing Python packages"
# $pipOutput = cmd.exe /c pip install -r requirements.txt
# Write-Host "Python packages installed"

Write-Host "Installation complete"
Start-Sleep -s 5
