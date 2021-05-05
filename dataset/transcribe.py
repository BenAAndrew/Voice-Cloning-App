import argparse
import torch
import os
import librosa
import torchaudio
import omegaconf

model, device, decoder = None, None, None


def load_audio(path):
    """
    Loads the audio from a given path into a tensor for transcription.

    Parameters
    ----------
    path : str
        Path to audio file

    Raises
    -------
    Exception
        If the audio file could not be loaded
    AssertionError
        If the audio file was empty
    """
    try:
        wav, _ = librosa.load(path, sr=16000)
    except Exception:
        raise Exception(f"Cannot load audio file {path}")

    assert len(wav) > 0, f"{path} wav file is empty"
    return torch.tensor([wav])


def transcribe(path):
    """
    Credit: https://github.com/snakers4/silero-models

    Transcribes a given audio file.
    Loads silero into a global variable to save time in future calls.

    Parameters
    ----------
    path : str
        Path to audio file

    Raises
    -------
    Exception
        If the audio file could not be loaded
    AssertionError
        If the audio file was not found or was empty

    Returns
    -------
    str
        Text transcription of audio file
    """
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


if __name__ == "__main__":
    """ Transcribe a clip """
    parser = argparse.ArgumentParser(description="Transcribe a clip")
    parser.add_argument("-i", "--input_path", help="Path to audio file", type=str, required=True)
    args = parser.parse_args()

    text = transcribe(args.input_path)
    print("Text: ", text)
