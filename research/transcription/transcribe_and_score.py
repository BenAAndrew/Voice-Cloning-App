import os
from transcribe_silero import transcribe_folder
from score_transcription import score_transcription

DATASET_PATH = "data"
OUTPUT_PATH = DATASET_PATH + "-score.csv"
AUDIO_PATH = os.path.join(DATASET_PATH, "wavs")
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")
TOTAL_SAMPLES = 1000

transcribe_folder(AUDIO_PATH, OUTPUT_PATH, TOTAL_SAMPLES)
print(score_transcription(METADATA_PATH, OUTPUT_PATH))
