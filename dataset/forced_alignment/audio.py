import wave
import collections

from webrtcvad import Vad

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH = 2
DEFAULT_FORMAT = (DEFAULT_RATE, DEFAULT_CHANNELS, DEFAULT_WIDTH)


def get_num_samples(pcm_buffer_size, audio_format=DEFAULT_FORMAT):
    """
    Credit: https://github.com/mozilla/DSAlign

    Gets number of samples in audio file.

    Parameters
    ----------
    pcm_buffer_size : int
        Size of audio PCM buffer
    audio_format : tuple
        Tuple containing the audio sample rate, channels & width

    Returns
    -------
    int
        Number of samples
    """
    _, channels, width = audio_format
    return pcm_buffer_size // (channels * width)


def get_pcm_duration(pcm_buffer_size, audio_format=DEFAULT_FORMAT):
    """
    Credit: https://github.com/mozilla/DSAlign

    Gets duration of audio file.

    Parameters
    ----------
    pcm_buffer_size : int
        Size of audio PCM buffer
    audio_format : tuple
        Tuple containing the audio sample rate, channels & width

    Returns
    -------
    float
        Audio duration
    """
    return get_num_samples(pcm_buffer_size, audio_format) / audio_format[0]


def read_frames(wav_file, frame_duration_ms=30, yield_remainder=False):
    """
    Credit: https://github.com/mozilla/DSAlign

    Read frames of audio file.

    Parameters
    ----------
    wav_file : wave
        Opened wav file using wave
    frame_duration_ms : int
        Frame duration in milliseconds
    yield_remainder : bool
        Whether to yield remaining audio frames

    Yields
    -------
    Audio frames
    """
    frame_size = int(DEFAULT_FORMAT[0] * (frame_duration_ms / 1000.0))
    while True:
        try:
            data = wav_file.readframes(frame_size)
            if not yield_remainder and get_pcm_duration(len(data), DEFAULT_FORMAT) * 1000 < frame_duration_ms:
                break
            yield data
        except EOFError:
            break


def read_frames_from_file(audio_path, audio_format=DEFAULT_FORMAT, frame_duration_ms=30, yield_remainder=False):
    """
    Credit: https://github.com/mozilla/DSAlign

    Read frames of audio file.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    audio_format : tuple
        Tuple containing the audio sample rate, channels & width
    frame_duration_ms : int
        Frame duration in milliseconds
    yield_remainder : bool
        Whether to yield remaining audio frames

    Yields
    -------
    Audio frames
    """
    audio = wave.open(audio_path, "r")
    for frame in read_frames(audio, frame_duration_ms=frame_duration_ms, yield_remainder=yield_remainder):
        yield frame


def vad_split(audio_frames, audio_format=DEFAULT_FORMAT, num_padding_frames=10, threshold=0.5, aggressiveness=3):
    """
    Credit: https://github.com/mozilla/DSAlign

    Splits audio into segments using Voice Activity Detection.

    Parameters
    ----------
    audio_frames : list
        List of audio frames
    audio_format : tuple
        Tuple containing the audio sample rate, channels & width
    num_padding_frames : int
        Number of frames to pad
    threshold : float
        Minimum threshold
    aggressiveness : int
        Aggressivess of VAD split

    Yields
    -------
    Audio segments (tuples containing number of frames, start time & end time))
    """

    sample_rate, channels, width = audio_format
    if channels != 1:
        raise ValueError("VAD-splitting requires mono samples")
    if width != 2:
        raise ValueError("VAD-splitting requires 16 bit samples")
    if sample_rate not in [8000, 16000, 32000, 48000]:
        raise ValueError("VAD-splitting only supported for sample rates 8000, 16000, 32000, or 48000")
    if aggressiveness not in [0, 1, 2, 3]:
        raise ValueError("VAD-splitting aggressiveness mode has to be one of 0, 1, 2, or 3")
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    vad = Vad(int(aggressiveness))
    voiced_frames = []
    frame_duration_ms = 0
    frame_index = 0
    for frame_index, frame in enumerate(audio_frames):
        frame_duration_ms = get_pcm_duration(len(frame), audio_format) * 1000
        if int(frame_duration_ms) not in [10, 20, 30]:
            raise ValueError("VAD-splitting only supported for frame durations 10, 20, or 30 ms")
        is_speech = vad.is_speech(frame, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > threshold * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > threshold * ring_buffer.maxlen:
                triggered = False
                yield b"".join(voiced_frames), frame_duration_ms * max(
                    0, frame_index - len(voiced_frames)
                ), frame_duration_ms * frame_index
                ring_buffer.clear()
                voiced_frames = []
    if len(voiced_frames) > 0:
        yield b"".join(voiced_frames), frame_duration_ms * (frame_index - len(voiced_frames)), frame_duration_ms * (
            frame_index + 1
        )
