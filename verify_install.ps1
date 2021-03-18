function Check-Installed {
    param (
        $app
    )
    $output = cmd.exe /c $app /w 2>&1
    return -Not ($output -like "*is not recognized*" -or $output -like "*was not found*" -or $output -like "Traceback*")
}

if(Check-Installed "nvcc --version"){
    Write-Host "CUDA: PASS"
} else {
    Write-Host "CUDA: FAIL"
}

# if(Check-Installed "python --version"){
#     Write-Host "Python: PASS"
# } else {
#     Write-Host "Python: FAIL"
# }

# if(Check-Installed "pip --version"){
#     Write-Host "pip: PASS"
# } else {
#     Write-Host "pip: FAIL"
# }

# if(Check-Installed 'python -c "import torch"'){
#     Write-Host "pytorch: PASS"
# } else {
#     Write-Host "pytorch: FAIL"
# }

if(Check-Installed "ffmpeg -version"){
    Write-Host "FFMPEG: PASS"
} else {
    Write-Host "FFMPEG: FAIL"
}

Start-Sleep -s 5
