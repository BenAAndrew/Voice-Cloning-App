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

<# Python install #>
if (-Not (Check-Installed "python")){
    Write-Host "Downloading Python 3.8..."
    Invoke-WebRequest -Uri "https://github.com/BenAAndrew/BenAAndrew.github.io/raw/master/resources/ai-voice-synthesis/python-3.8.7.exe" -OutFile "python-3.8.7.exe"
    Write-Host "Installing Python 3.8"
    $pyInstallOutput = cmd.exe /c python-3.8.7.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    Remove-Item -Path python-3.8.7.exe
    if(-Not (Check-Installed "python")){
        Error "Python did not install successfully"
    }
    Write-Host "Installed Python 3.8"
} else {
    Write-Host "Python already installed"
}

if(-Not (Check-Installed "pip")){
    Error "pip has not been installed successfully"
}

<# Pytorch install #>
$pytorchOutput = cmd.exe /c pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
Write-Host "Pytorch installed"

<# FFMPEG install #>
if (-Not (Check-Installed "ffmpeg")){
    Write-Host "Downloading FFMPEG..."
    Invoke-WebRequest -Uri "https://github.com/BenAAndrew/BenAAndrew.github.io/raw/master/resources/ai-voice-synthesis/ffmpeg.zip" -OutFile "ffmpeg.zip"
    Write-Host "Installing FFMPEG..."
    Expand-Archive -Path ffmpeg.zip -DestinationPath C:\
    [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ffmpeg\bin", "User")
    Remove-Item -Path ffmpeg.zip
    if(-Not (Check-Installed "ffmpeg")){
        Error "FFMPEG did not install successfully"
    }
    Write-Host "Installed FFMPEG" 
} else {
    Write-Host "FFMPEG installed"
}

<# Dependency install #>
Write-Host "Installing Python packages"
$pipOutput = cmd.exe /c pip install -r requirements.txt
Write-Host "Python packages installed"

Write-Host "Installation complete"
Start-Sleep -s 5
