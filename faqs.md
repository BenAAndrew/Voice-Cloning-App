# FAQ's

## "I cannot train as it says 'CUDA is not available on my system' but I have an NVIDIA GPU"

The CUDA runtime libraries are included with the NVIDIA drivers so as long as you have an up-to-date and correctly installed driver you should be OK. If you run `nvidia-smi` in terminal it should show your CUDA version (ensure it is version 11+).

If you have manually installed pytorch it is possible that you have installed the cpu only version. Please uninstall and reinstall pytorch with the CUDA version (use `--no-cache-dir --force-reinstall` to ensure it doesn't use previously installed cache)

## "If I stop training will I lose my progress?"

No. Training will automatically continue from the latest checkpoint so you can stop at any time and it will carry on from where it left off next time you start training.

## "How much data do I need to make a good voice?"

For this model I would recommend at least 2 hours of data and 4000 epochs. Every dataset is different and will produce different results so the scoring on the training page is a very rough guess. Using a pretrained model for transfer learning may improve results as well.

## "Can I generate other languages?"
Unfortunately this is not currently supported but may be added in the future.

## "Can I make changes to the project?"
Everyone is welcome to open pull requests and suggest changes by raising an "enhancement" issue.

Please read [How to make changes](maintenance.md) before making a change
