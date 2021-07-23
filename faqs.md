# FAQ's

## "I cannot train as it says 'CUDA is not available on my system' but I have an NVIDIA GPU"
The CUDA runtime libraries are included with the NVIDIA drivers so as long as you have an up-to-date and correctly installed driver you should be OK. If you run `nvidia-smi` in terminal it should show your CUDA version (ensure it is version 11+).

If you have manually installed pytorch it is possible that you have installed the cpu only version. Please uninstall and reinstall pytorch with the CUDA version (use `--no-cache-dir --force-reinstall` to ensure it doesn't use previously installed cache)

## "If I stop training will I lose my progress?"
No. Training will automatically continue from the latest checkpoint so you can stop at any time and it will carry on from where it left off next time you start training.

## "How much data do I need to make a good voice?"
For this model I would recommend at least 2 hours of data and training for 2000 epochs with transfer learning.

Every dataset is different and will produce different results so the scoring on the training page is a very rough guess.

## "Can I generate other languages?"
Yes, but you will need to add a deepspeech voice of that language in settings. These can be found on sites such as [coqui](https://coqui.ai/models).

Some of these perform better than others so quality may vary.

## "Can I make changes to the project?"
Everyone is welcome to open pull requests or suggest changes by raising an "enhancement" issue.

Please read [How to make changes](maintenance.md) before making a change

## "The produced voice is poor quality. What have I done wrong?"
Check the following:

**Dataset**

1. How big is your dataset? Do you have at least 1 hour of clips? If not this will limit the potential quality of the voice
2. How clear are the clips? Try listening to 5 random samples in the wavs folder and make sure the audio is clear and free of background noise/ other peoples voices
3. Are the clips correctly labelled? Try listening to 5 random samples and compare what is said to its label in metadata.csv. If samples are being mislabelled they will make the quality of the voice worse. You can typically improve the label accuracy by increasing the minimum confidence score in the dataset step
4. Do your labels have punctuation? This is essential for the voice to work correctly

**Training**

5. How many epochs have you trained for? This depends on the size of your dataset but you should see the loss decrease to a certain point where it no longer improves significantly. Test results frequently so you can hear how good the voice is at different amounts of epochs. Also be aware you can over-train if you train for too long relative to the size of your dataset
6. Did you use transfer learning? This will give your model a better starting point and should lead it to higher quality results sooner into training

**Synthesis**

7. Have you made your sentence too long/short? Ensure your sentence produces audio of around 2-8 seconds in length
8. Have you tried substituting some words? Certain voices struggle with certain words but you can usually fix this by rewriting the word so that is is spelt more like how it sounds
