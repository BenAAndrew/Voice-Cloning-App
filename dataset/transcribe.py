import torch
import os
import librosa
import torchaudio
import omegaconf

model, device, decoder = None, None, None


def load_audio(path):
    try:
        wav, _ = librosa.load(path, sr=16000)
    except Exception:
        raise Exception(f"Cannot load audio file {path}")

    max_seqlength = max(len(wav), 12800)
    data = torch.zeros(1, max_seqlength)
    data[0] = torch.tensor(wav)
    return data


def transcribe(path):
    global model, device, decoder, read_batch, split_into_batches, prepare_model_input
    assert os.path.isfile(path), f"{path} not found. Cannot transcribe"

    if not model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, decoder, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models", model="silero_stt", language="en", device=device
        )

    data = load_audio(path)
    data = data.to(device)
    output = model(data)

    for example in output:
        return decoder(example.cpu())
