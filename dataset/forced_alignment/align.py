import os
import logging

import sys
sys.path.append( os.path.abspath("../../"))

from dataset.forced_alignment.audio import DEFAULT_RATE, read_frames_from_file, vad_split
from dataset.audio_processing import cut_audio
from dataset.transcribe import transcribe


logging.getLogger().setLevel(logging.INFO)
audio_vad_threshold = 0.5
audio_vad_aggressiveness = 3
audio_vad_frame_length = 30
audio_vad_padding = 10


def enweight(items, direction=0):
    """
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


def get_segments(audio_path, output_path):
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


def process_segments(audio_path, output_path, segments, min_length, max_length, logging=logging):
    logging.info("Generating segments...")
    samples = []
    total = len(segments)
    index = 0
    for i in range(total):
        segment = segments[i]
        _, time_start, time_end = segment
        time_length = time_end - time_start

        if time_length >= min_length and time_length <= max_length:
            name = cut_audio(audio_path, int(time_start), int(time_end), output_path)
            clip_path = os.path.join(output_path, name)
            transcript = transcribe(clip_path)
            if transcript:
                samples.append({"index": index, "start": time_start, "end": time_end, "name": name, "transcript": transcript.strip()})
                index += 1

        logging.info(f"Progress - {i+1}/{total}")
    return samples


def split_match(fragments, search, start=0, end=-1):
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
        match_start, match_end, sws_score, match_substitutions = match
        if sws_score > (n - 1) / (2 * n):
            fragment["match-start"] = match_start
            fragment["match-end"] = match_end
            fragment["sws"] = sws_score
            for f in split_match(fragments[0:index], search, start=start, end=match_start):
                yield f
            yield fragment
            for f in split_match(fragments[index + 1 :], search, start=match_end, end=end):
                yield f
            return
    for _, _ in weighted_fragments:
        yield None
