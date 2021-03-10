import os
import json
import logging
import argparse
from pydub import AudioSegment

from dataset.forced_alignment.search import FuzzySearch
from dataset.forced_alignment.audio import DEFAULT_RATE, read_frames_from_file, vad_split
from dataset.transcribe import stt


logging.getLogger().setLevel(logging.INFO)

audio_vad_threshold = 0.5
audio_vad_aggressiveness = 3
audio_vad_frame_length = 30
audio_vad_padding = 10
stt_min_duration = 100
stt_max_duration = 100000
align_max_candidates = 10
align_candidate_threshold = 0.92
align_match_score = 100
align_mismatch_score = -100
align_gap_score = -100
MIN_SWS = 0.5


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


def get_segments(audio_path):
    model_format = (DEFAULT_RATE, 1, 2)
    frames = read_frames_from_file(audio_path, model_format, audio_vad_frame_length)
    segments = vad_split(
        frames,
        model_format,
        num_padding_frames=audio_vad_padding,
        threshold=audio_vad_threshold,
        aggressiveness=audio_vad_aggressiveness,
    )
    samples = []
    for segment in segments:
        _, time_start, time_end = segment
        time_length = time_end - time_start
        if time_length > stt_min_duration and time_length < stt_max_duration:
            samples.append((time_start, time_end))
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


def align(audio_path, script_path, aligned_path, logging=logging):
    logging.info(f"Loading audio from {audio_path}...")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(22050)
    audio = audio.set_channels(1)

    logging.info(f"Loading script from {script_path}...")
    with open(script_path, "r", encoding="utf-8") as script_file:
        clean_text = script_file.read().lower()

    logging.info("Fetching segments...")
    samples = get_segments(audio_path)
    logging.info("Transcribing segments...")
    fragments = []
    total_samples = len(samples)
    index = 0
    for i in range(total_samples):
        sample = samples[i]
        time_start, time_end = sample
        transcript = stt(audio, sample)
        if transcript:
            fragments.append({"index": index, "start": time_start, "end": time_end, "transcript": transcript.strip()})
            index += 1

        logging.info(f"Progress - {i+1}/{total_samples}")
    logging.info(f"Created {len(fragments)} fragments")

    logging.info("Searching text for matching fragments...")
    search = FuzzySearch(
        clean_text,
        max_candidates=align_max_candidates,
        candidate_threshold=align_candidate_threshold,
        match_score=align_match_score,
        mismatch_score=align_mismatch_score,
        gap_score=align_gap_score,
    )
    matched_fragments = split_match(fragments, search)
    matched_fragments = list(filter(lambda f: f is not None, matched_fragments))
    logging.info(f"Matched {len(matched_fragments)} fragments")

    result_fragments = []
    for fragment in matched_fragments:
        if (
            fragment["sws"] > MIN_SWS
            and "match-start" in fragment
            and "match-end" in fragment
            and fragment["match-end"] - fragment["match-start"] > 0
        ):
            fragment_matched = clean_text[fragment["match-start"] : fragment["match-end"]]
            if fragment_matched:
                fragment["aligned"] = fragment_matched.strip().replace("\n", " ")
                for key in ["match-start", "match-end", "index"]:
                    del fragment[key]
                result_fragments.append(fragment)

    logging.info(f"Produced {len(result_fragments)} final fragments")
    with open(aligned_path, "w", encoding="utf-8") as result_file:
        result_file.write(json.dumps(result_fragments, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Force align speech data with a transcript.")
    parser.add_argument("--audio", type=str, help="Path to speech audio file")
    parser.add_argument("--script", type=str, help="Path to original transcript (plain text or .script file)")
    parser.add_argument("--aligned", type=str, help="Path to aligned file")
    args = parser.parse_args()

    align(args.audio, args.script, args.aligned)
