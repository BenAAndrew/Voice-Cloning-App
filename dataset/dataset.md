# Building the dataset
To generate a dataset for voice synthesis, we need a text & audio source. This can be extracted from a source such as an audiobook or audio with subtitle labelling.

We can the use this source to produce a dataset of labelled audio clips of the target.

## Option A: Audiobook
Firstly we need to get an audiobook and extract it's audio & text. The two best sources I've found for audiobooks are [Audible](https://www.audible.co.uk/) and [LibriVox](https://librivox.org/). 

### Audible
Audible books are licenced by audible and need to be purchased before use. For this project you will need to look for [Kindle books with audio narration](https://www.amazon.co.uk/Kindle-Books-with-Audio-Companions/b?ie=UTF8&node=5123320031). 

Once you purchase an audiobook & matching kindle book, steps for extracting it can be found in a link under the dataset step of the app or you can follow the [youtube tutorial](https://www.youtube.com/watch?v=oS5VMxbhREE). 

### LibriVox
Whilst LibriVox is open source, it's quality is generally less consistent. However, if you find a book with audio and text you can use it just the same as the audible method.

## Option B: Subtitle
If you have an audiofile with matching subtitle file you can also use this as a source.
You may have a video file with embedded subtitles in which case you need to extract the audio & subtitles from this first.

## Generate dataset
Once we have the text and audio source (from either of the options above), we need to produce snippets of speech with labels. To do this you can run `create_dataset.py`.

For audiobook/plain text:
``` python create_dataset.py --audio_path book.wav --text_path book.txt --output_path wavs --label_path metadata.csv```

For subtitles:
``` python create_dataset.py --audio_path book.wav --text_path sub.srt --output_path wavs --label_path metadata.csv```

### Optional: Analyse dataset
To see a breakdown of the key stats of your dataset run

``` python analysis.py --wavs wavs --metadata metadata.csv ```
