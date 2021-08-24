import os
import logging

from dataset.forced_alignment.audio import DEFAULT_RATE, read_frames_from_file, vad_split
from dataset.audio_processing import get_timestamp, cut_audio


logging.getLogger().setLevel(logging.INFO)
audio_vad_threshold = 0.5
audio_vad_aggressiveness = 3
audio_vad_frame_length = 30
audio_vad_padding = 10


def enweight(items, direction=0):
    """
    Credit: https://github.com/mozilla/DSAlign

    Enumerates all entries together with a positional weight value.
    The positional weight progresses quadratically.
    :param items: Items to enumerate
    :param direction: Order of assigning positional weights to N-grams:
        direction < 0: Weight of first N-gram is 1.0 and of last one 0.0
        direction > 0: Weight of first N-gram is 0.0 and of last one 1.0
        direction == 0: Weight of center N-gram(s) near or equal 0, weight of first and last N-gram 1.0
    :return: Produces (object, float) tuples representing the enumerated item
             along with its assigned positional weight value
    """
    items = list(items)
    direction = -1 if direction < 0 else (1 if direction > 0 else 0)
    n = len(items) - 1
    if n < 1:
        if n == 0:
            yield items[0], 1
        raise StopIteration
    for i, item in enumerate(items):
        c = (i + n * (direction - 1) / 2) / n
        yield item, c * c * (4 - abs(direction) * 3)


def get_segments(audio_path):
    """
    Credit: https://github.com/mozilla/DSAlign

    Generates segments for an given audio file.

    Parameters
    ----------
    audio_path : str
        Path to audio file (must be converted to a sample rate of 16000)

    Returns
    -------
    list
        List of segments (tuples containing number of frames, start time & end time)
    """
    model_format = (DEFAULT_RATE, 1, 2)
    frames = read_frames_from_file(audio_path, model_format, audio_vad_frame_length)
    segments = vad_split(
        frames,
        model_format,
        num_padding_frames=audio_vad_padding,
        threshold=audio_vad_threshold,
        aggressiveness=audio_vad_aggressiveness,
    )
    return [segment for segment in segments]


def process_segments(audio_path, transcription_model, output_path, segments, min_length, max_length, logging=logging):
    """
    Generates audio clips and reduces segments to only valid ones.
    This includes removing segements which are too long, too short or cannot be transcribed.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    transcription_model : TranscriptionModel model
        TranscriptionModel transcription model
    output_path : str
        Path to save clips to
    segments : list
        List of segments produced in get_segments
    min_length : int
        Minimum length of a clip (in milliseconds)
    max_length : int
        Maximum length of a clip (in milliseconds)
    logging : logging (optional)
        Logging object to write progress to

    Returns
    -------
    list
        List of samples (dictionaries containing clip index, start, end, name & transcript)
    """
    logging.info("Generating segments...")
    samples = []
    total = len(segments)
    for i in range(total):
        segment = segments[i]
        _, time_start, time_end = segment
        duration = time_end - time_start

        if duration >= min_length and duration <= max_length:
            start = get_timestamp(int(time_start))
            end = get_timestamp(int(time_end))
            name = cut_audio(audio_path, start, end, output_path)
            clip_path = os.path.join(output_path, name)

            try:
                transcript = transcription_model.transcribe(clip_path)
            except:
                logging.info(f"Could not transcribe {clip_path}")
                transcript = None

            if transcript:
                samples.append(
                    {
                        "name": name,
                        "start": start,
                        "end": end,
                        "duration": duration / 1000,
                        "transcript": transcript.strip(),
                    }
                )

        logging.info(f"Progress - {i+1}/{total}")
    return samples


def split_match(fragments, search, start=0, end=-1):
    """
    Credit: https://github.com/mozilla/DSAlign

    Matches fragments to text file.

    Parameters
    ----------
    fragments : list
        List of fragments to match
    search : FuzzySearch
        Source text object
    start : int
        Start index
    end : int
        End index

    Yields
    -------
    Matching fragments with match start, end & sws (score)
    """
    n = len(fragments)
    if n < 1:
        return
    elif n == 1:
        weighted_fragments = [(0, fragments[0])]
    else:
        # so we later know the original index of each fragment
        weighted_fragments = enumerate(fragments)
        # assigns high values to long statements near the center of the list
        weighted_fragments = enweight(weighted_fragments)
        weighted_fragments = map(lambda fw: (fw[0], (1 - fw[1]) * len(fw[0][1]["transcript"])), weighted_fragments)
        # fragments with highest weights first
        weighted_fragments = sorted(weighted_fragments, key=lambda fw: fw[1], reverse=True)
        # strip weights
        weighted_fragments = list(map(lambda fw: fw[0], weighted_fragments))

    for index, fragment in weighted_fragments:
        match = search.find_best(fragment["transcript"], start=start, end=end)
        match_start, match_end, score = match
        if score > (n - 1) / (2 * n):
            fragment["match-start"] = match_start
            fragment["match-end"] = match_end
            fragment["score"] = score
            for f in split_match(fragments[0:index], search, start=start, end=match_start):
                yield f
            yield fragment
            for f in split_match(fragments[index + 1 :], search, start=match_end, end=end):
                yield f
            return
    for _, _ in weighted_fragments:
        yield None
