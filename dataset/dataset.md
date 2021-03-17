# Building the dataset
To generate a dataset for voice synthesis, we need labelled audio clips of the target. A good starting point to get this data from an audiobook as the audio is clear and we can typically extract several hours of data.

Steps:
1. [**Gather audio/text from an audiobook**](#gather-audiotext)
    1. [Audible method](#audible)
    2. [Librivox method](#librivox)
2. [**Force align the text and audio**](#force-align-text-and-audio)
3. [**Generate clips**](#generate-clips)
4. [Analyse dataset (optional)](#optional-analyse-dataset)


## Gather audio/text
Firstly we need to get an audiobook and extract it's audio & text. The two best sources I've found for audiobooks are [Audible](https://www.audible.co.uk/) and [LibriVox](https://librivox.org/). 

### Audible
Audible books are licenced by audible and need to be purchased before use. For this project you will need to look for [Kindle books with audio narration](https://www.amazon.co.uk/Kindle-Books-with-Audio-Companions/b?ie=UTF8&node=5123320031). 

Once you get one, we need to convert the Audible AAX audio into a WAV file. To do this, find where the Audible app has saved this file and then use a tool such as [AaxAudioConverter](https://github.com/audiamus/AaxAudioConverter) to convert it.

Then extract the text using the chrome extension in the 'extension' folder. Steps on how to use this can be found in step 1 of the app.

### LibriVox
Whilst LibriVox is open source, it's quality is generally less consistent. However, if you find a book with audio and text you can use it just the same as the audible method.

## Force align text and audio
Once we have the text and audio of an audiobook, we need to align the two. To do this you can run the `align.py` script in 'forced_alignment'

``` python forced_alignment/align.py --audio book.wav --script book.txt --aligned aligned.json ```

## Generate clips

Using `clip_generator.py` we can take the JSON from forced alignment and the audio to produce snippets of speech with labels.

Example usage:

``` python clip_generator.py --audio_path book.wav --forced_alignment_path align.json --output_path wavs --label_path metadata.csv ```

## Optional: Analyse dataset

To see a breakdown of the key stats of your dataset run

``` python analysis.py --wavs wavs --metadata metadata_clean.csv ```
