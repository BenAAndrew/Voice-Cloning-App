import torch
from glob import glob

model, device, decoder, read_batch, split_into_batches, prepare_model_input = None, None, None, None, None, None


def transcribe(path):
    global model, device, decoder, read_batch, split_into_batches, prepare_model_input

    if not model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, decoder, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-models", model="silero_stt", language="en", device=device
        )
        (read_batch, split_into_batches, _, prepare_model_input) = utils

    audio = glob(path)
    batches = split_into_batches(audio, batch_size=10)
    data = prepare_model_input(read_batch(batches[0]), device=device)

    output = model(data)
    for example in output:
        return decoder(example.cpu())


def stt(audio, sample):
    time_start, time_end = sample
    snippet = audio[int(time_start) : int(time_end)]
    snippet.export("temp.wav", format="wav")
    transcript = transcribe("temp.wav")
    return " ".join(transcript.split())
