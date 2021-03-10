# Synthesising the voice
With trained model weights we can now implement the TTS app for our generated voice model.

## Arguments
- **model_path**: The path to your generated model
- **waveglow_model_path**: The path to your waveglow model ([default found here](https://github.com/BenAAndrew/Voice-Cloning-Assets/raw/main/waveglow_256channels_universal_v5.pt))
- **text**: Text you wish to synthesize
- **graph_output_path (optional)**: Path to save alignment graph to
- **audio_output_path (optional)**: Path to save generated audio to

## How to run
`python synthesize.py -m checkpoint_500000 -w waveglow_256channels_universal_v5.pt -t "Hello everyone, how are you?" -g graph.png -a audio.wav
