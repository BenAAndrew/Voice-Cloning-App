import torch
import os
import random
from glob import glob
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, decoder, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-models", model="silero_stt", language="en", device=device
)
(read_batch, split_into_batches, read_audio, prepare_model_input) = utils


def transcribe(path):
    audio = glob(path)
    batches = split_into_batches(audio, batch_size=10)
    data = prepare_model_input(read_batch(batches[0]), device=device)

    output = model(data)
    for example in output:
        return decoder(example.cpu())


def transcribe_folder(path, output_path, n_samples):
    files = random.sample(os.listdir(path), n_samples)
    sentences = {}

    for f in tqdm(files):
        sentences[f] = transcribe(os.path.join(path, f))

    with open(output_path, "w") as f:
        for key, value in sentences.items():
            f.write(f"{key}|{value}\n")
