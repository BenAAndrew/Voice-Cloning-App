# Maintenance
This guide is designed to help with understanding and maintaining this project.
If you wish to make changes to a feature, please read this first.

## How to get involved
Everyone is welcome to make modifications to the project and propose their changes in pull requests. Please make sure that you understand and have tested your changes before sharing.

## Project Structure
- [Application](#application)
- [Dataset](#dataset)
- [Training](#training)
- [Synthesis](#synthesis)

## Application
The frontend application is build using `flask` and `flask-socketio`.

`main.py` starts the app and opens the browser but the majority of the app is handled in the `application` folder.

`check_ffmpeg.py` checks that an ffmpeg install is found and if not will install. For linux it runs the install command. For windows it downloads and extracts an ffmpeg zip folder. This could fail if the URL is no longer supported, so should be wrapped in the executable if possible.

`views.py` contains all of the endpoints. Depending on the complexity of the task it may call `start_progress_thread`. This takes a function and runs it in a background thread using `flask-socketio`. These tasks must do the following to be supported by  `start_progress_thread`:
- The function must write info to the passed logger (called `logging`)
- To update the progress bar it must write a log in the format "Progress - x/y" where x is the current iteration and y is the total iterations (i.e. "Progress - 25/100")
- To pin a message it must write a log in the format "Status - text" where text is the message (i.e. "Status - Score is 0.5")

The Frontend handles messsages from the thread in `application.js`.
If an error occurs, this will be sent in an "error" message to the frontend. When complete, the handler will send a "done" message to the frontend which shows the "next" button.

Please note: The function called inside this thread cannot create other threads or the application may crash.

## Dataset
The dataset builder uses a range of libraries including `pydub`, `librosa`, `torch`, `wave` and `webrtcvad`.

The main entry for the builder is `dataset/create_dataset.py`. It does 3 things:
1. Converts the audio to consistent format using `FFmpeg` in `audio_processing.py` 
2. Generates clips using `clip_generator.py`
3. Generates an info file using `analysis.py`

The forced alignment process used in `clip_generator.py` can be found in the `forced_alignment` folder and is based on https://github.com/mozilla/DSAlign. This library is able to take the source audio and text and split into clips. 

It uses `transcribe.py` which uses https://github.com/snakers4/silero-models to convert speech-to-text and will delete clips which do not meet a minimum similarity score.

`clip_generator.py` will also remove clips with an invalid duration and save the resulting filenames and text to a metadata file (typically called "metadata.csv") in the format filename|text (i.e. "clip_1.wav|Hello, how are you?").

`extend_existing_dataset.py` uses `clip_generator.py` to extend an existing dataset, and adds a suffix to filenames to differentiate sources.

## Training
The training script implements a modified version https://github.com/NVIDIA/tacotron2.

Found in `train.py` it add a few additions the existing project did not have:
1. It ensures CUDA is enabled before starting. This is required for `torch` to use the GPU and is essential for this model.
2. It automatically calculates what batch size & learning rate to use depending on the available GPU memory. This is only a conservative estimate so can be tweaked, but is a useful starting point for inexperienced users
3. It can automatically search the output model folder to find a checkpoint to start training from. This is what enables the easy start-stop functionality, so that users can continue training from where they left off
4. It can enable early-stopping which will stop training if the loss over the last 10 checkpoints has not sufficently decreased (minimum loss reached) 

## Synthesis
The synthsis script implements https://github.com/NVIDIA/waveglow.

The synthesis process is implemented in `synthesize.py`. It firstly loads the feature predictor model (from training) and a pretrained waveglow model. It then cleans the text and infers the results. Audio & an alignment graph can be produced from this.

In the app `synonyms.py` is also used to suggest synonyms for poorly synthsised words. It uses the `trainscribe.py` script to estimate which words were non-audible, and the `nltk` library to list synonyms.
