# Synthesising the voice
With trained model weights we can now implement the TTS app for our generated voice model.

## Arguments
- **model_path**: The path to your generated model
- **vocoder_model_path**: The path to your vocoder model ([default hifigan model found here](https://drive.google.com/file/d/1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW/view?usp=sharing))
- **hifigan_config_path**: The path to your hifigan config ([default higigan config found here](https://drive.google.com/file/d/1pAB2kQunkDuv6W5fcJiQ0CY8xcJKB22e/view?usp=sharing))
- **text**: Text you wish to synthesize
- **graph_output_path (optional)**: Path to save alignment graph to
- **audio_output_path (optional)**: Path to save generated audio to
- **silence_padding (optional)** : Seconds of silence to seperate each clip by with multi-line synthesis (default is 0.15)
- **sample_rate (optional)** : Audio sample rate (default is 22050)
- **max_decoder_steps (optional)** : Max decoder steps controls sequence length and memory usage during inference. Increasing this will use more memory but may allow for longer sentences. (default is 1000)

## How to run
`python synthesize.py -m checkpoint_500000 -vm g_02500000 -hc config.json -t "Hello everyone, how are you?" -g graph.png -a audio.wav`
