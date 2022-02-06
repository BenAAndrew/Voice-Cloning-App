import argparse
import os
import logging
import json
import unicodedata
import pysrt
from pydub import AudioSegment
from datetime import datetime

from dataset import MIN_LENGTH, MAX_LENGTH, MIN_CONFIDENCE, CHARACTER_ENCODING
from dataset.utils import similarity, get_invalid_characters
import dataset.forced_alignment.align as align
from dataset.forced_alignment.search import FuzzySearch
from dataset.forced_alignment.audio import DEFAULT_RATE
from dataset.audio_processing import change_sample_rate, cut_audio, add_silence
from training import DEFAULT_ALPHABET, PUNCTUATION


def clip_combiner(audio_path, output_path, clips, max_length):
    """
    Combines clips to make them as long as possible without exceeding max length.

    Parameters
    ----------
    audio_path : str
        Path to audio file (must have been converted using convert_audio)
    output_path : str
        Path to save audio clips to
    clips : list
        List of current clips
    max_length : float (optional)
        Maximum duration of a clip in seconds

    Returns
    -------
    (list, list)
        List of clips and clip lengths in seconds
    """

    def _get_duration(start, end):
        """Gets the duration in seconds between two string timestamps"""
        return (datetime.strptime(end, "%H:%M:%S.%f") - datetime.strptime(start, "%H:%M:%S.%f")).total_seconds()

    def _join_text(lines):
        """Joins list of lines with comma seperation"""
        return " ".join(
            [
                line + "," if not line[-1] in PUNCTUATION and i != len(lines) - 1 else line
                for i, line in enumerate(lines)
            ]
        )

    def _combine_clip(combined_clip, audio_path, output_path):
        """Combines multiple clips to produce one new clip (or returns existing if list contains only one clip)"""
        if len(combined_clip) > 1:
            start = combined_clip[0]["start"]
            end = combined_clip[-1]["end"]
            duration = _get_duration(start, end)
            filename = cut_audio(audio_path, start, end, output_path)
            return {
                "name": filename,
                "start": start,
                "end": end,
                "duration": duration,
                "transcript": _join_text([clip["transcript"] for clip in combined_clip]),
                "text": _join_text([clip["text"] for clip in combined_clip]),
                "score": sum([clip["score"] for clip in combined_clip]) / len(combined_clip),
            }, duration
        else:
            return combined_clip[0], combined_clip[0]["duration"]

    output_clips = []
    clip_lengths = []
    combined_clip = []
    for clip in clips:
        if combined_clip:
            if _get_duration(combined_clip[0]["start"], clip["end"]) > max_length:
                new_clip, duration = _combine_clip(combined_clip, audio_path, output_path)
                output_clips.append(new_clip)
                clip_lengths.append(duration)
                combined_clip = []
        combined_clip.append(clip)

    # Add final combined clip
    if combined_clip:
        new_clip, duration = _combine_clip(combined_clip, audio_path, output_path)
        output_clips.append(new_clip)
        clip_lengths.append(duration)
    return output_clips, clip_lengths


def generate_clips_from_textfile(
    audio_path,
    text,
    transcription_model,
    output_path,
    logging=logging,
    min_length=MIN_LENGTH,
    max_length=MAX_LENGTH,
    min_confidence=MIN_CONFIDENCE,
):
    """
    Generates clips from plain text file.

    Parameters
    ----------
    audio_path : str
        Path to audio file (must have been converted using convert_audio)
    text : str
        Source text
    transcription_model : TranscriptionModel
        Transcription model
    output_path : str
        Path to save audio clips to
    logging : logging (optional)
        Logging object to write logs to
    min_length : float (optional)
        Minimum duration of a clip in seconds
    max_length : float (optional)
        Maximum duration of a clip in seconds
    min_confidence : float (optional)
        Minimum confidence score to generate a clip for

    Returns
    -------
    (list, list)
        List of clips and clip lengths in seconds
    """
    logging.info("Searching text for matching fragments...")
    search = FuzzySearch(text)

    logging.info("Changing sample rate...")
    converted_audio_path = change_sample_rate(audio_path, DEFAULT_RATE)

    # Produce segments
    logging.info("Fetching segments...")
    segments = align.get_segments(converted_audio_path)

    # Match with text
    logging.info("Matching segments...")
    min_length_ms = min_length * 1000
    max_length_ms = max_length * 1000
    processed_segments = align.process_segments(
        audio_path, transcription_model, output_path, segments, min_length_ms, max_length_ms, logging
    )
    matched_segments = align.split_match(processed_segments, search)
    matched_segments = list(filter(lambda f: f is not None, matched_segments))
    logging.info(f"Matched {len(matched_segments)} segments")

    result_fragments = []
    unlabelled_fragments = []
    clip_lengths = []
    for fragment in matched_segments:
        if (
            fragment["transcript"]
            and fragment["score"] >= min_confidence
            and "match-start" in fragment
            and "match-end" in fragment
            and fragment["match-end"] - fragment["match-start"] > 0
        ):
            fragment_matched = text[fragment["match-start"] : fragment["match-end"]]
            if fragment_matched:
                fragment["text"] = fragment_matched
                clip_lengths.append(fragment["duration"])
                result_fragments.append(fragment)
        else:
            unlabelled_fragments.append(fragment["name"])

    os.remove(converted_audio_path)
    return result_fragments, unlabelled_fragments, clip_lengths


def generate_clips_from_subtitles(
    audio_path,
    subs,
    transcription_model,
    output_path,
    logging=logging,
    min_length=MIN_LENGTH,
    max_length=MAX_LENGTH,
    min_confidence=MIN_CONFIDENCE,
):
    """
    Generates clips from subtitles.

    Parameters
    ----------
    audio_path : str
        Path to audio file (must have been converted using convert_audio)
    subs : list
        List of pysrt subtitle objects
    transcription_model : TranscriptionModel
        Transcription model
    output_path : str
        Path to save audio clips to
    logging : logging (optional)
        Logging object to write logs to
    min_length : float (optional)
        Minimum duration of a clip in seconds
    max_length : float (optional)
        Maximum duration of a clip in seconds
    min_confidence : float (optional)
        Minimum confidence score to generate a clip for

    Returns
    -------
    (list, list)
        List of clips and clip lengths in seconds
    """
    logging.info("Loading subtitles...")
    total = len(subs)
    logging.info(f"{total} subtitle lines detected...")

    result_fragments = []
    unlabelled_fragments = []
    clip_lengths = []
    for i, sub in enumerate(subs):
        duration = sub.duration.seconds + (sub.duration.milliseconds / 1000)
        if duration >= min_length and duration <= max_length:
            start = sub.start.to_time().strftime("%H:%M:%S.%f")
            end = sub.end.to_time().strftime("%H:%M:%S.%f")
            filename = cut_audio(audio_path, start, end, output_path)
            clip_path = os.path.join(output_path, filename)

            try:
                transcript = transcription_model.transcribe(clip_path)
            except Exception as e:
                logging.info(f"Could not transcribe {clip_path}: {e}")
                transcript = None

            if transcript:
                text = sub.text.strip().replace("\n", " ")
                score = similarity(transcript, text)
                if score >= min_confidence:
                    result_fragments.append(
                        {
                            "name": filename,
                            "start": start,
                            "end": end,
                            "duration": duration,
                            "transcript": transcript,
                            "text": text,
                            "score": score,
                        }
                    )
                    clip_lengths.append(duration)
            else:
                unlabelled_fragments.append(filename)
        logging.info(f"Progress - {i+1}/{total}")

    return result_fragments, unlabelled_fragments, clip_lengths


def clip_generator(
    audio_path,
    script_path,
    transcription_model,
    forced_alignment_path,
    output_path,
    unlabelled_path,
    label_path,
    logging=logging,
    symbols=DEFAULT_ALPHABET,
    min_length=MIN_LENGTH,
    max_length=MAX_LENGTH,
    silence_padding=0.1,
    min_confidence=MIN_CONFIDENCE,
    combine_clips=True,
):
    """
    Generates dataset clips & label file.
    Also combines clips, adds silence, produces metadata/info & does cleanup.

    Parameters
    ----------
    audio_path : str
        Path to audio file (must have been converted using convert_audio)
    script_path : str
        Path to source text
    transcription_model : TranscriptionModel
        Transcription model
    forced_alignment_path : str
        Path to save alignment JSON to
    output_path : str
        Path to save audio clips to
    unlabelled_path : str
        Path to save unlabelled audio clips to
    label_path : str
        Path to save label file to
    logging : logging (optional)
        Logging object to write logs to
    symbols : list (optional)
        List of valid symbols
    min_length : float (optional)
        Minimum duration of a clip in seconds
    max_length : float (optional)
        Maximum duration of a clip in seconds
    silence_padding : float (optional)
        Padding of silence at the end of each clip in seconds
    min_confidence : float (optional)
        Minimum confidence score to generate a clip for
    combine_clips : bool (optional)
        Whether to combine clips to make them longer

    Raises
    -------
    AssertionError
        If given paths are invalid or clips could not be produced

    Returns
    -------
    list
        List of clip lengths in seconds
    """
    assert not os.path.isdir(
        output_path
    ), "Output directory already exists. Did you mean to use 'Extend existing dataset'?"
    assert os.path.isfile(audio_path), "Audio file not found"
    assert os.path.isfile(script_path), "Script file not found"
    assert audio_path.endswith(".wav"), "Must be a WAV file"

    os.makedirs(output_path, exist_ok=False)
    os.makedirs(unlabelled_path, exist_ok=False)

    # Validate text
    is_subtitle = script_path.endswith(".srt")
    logging.info(f"Loading {script_path}...")

    if is_subtitle:
        subs = pysrt.open(script_path)
        text = " ".join([sub.text for sub in subs])
    else:
        with open(script_path, "r", encoding=CHARACTER_ENCODING) as script_file:
            text = script_file.read()

    text = text.lower().strip().replace("\n", " ").replace("  ", " ")
    invalid_chars = get_invalid_characters(text, symbols)
    assert (
        not invalid_chars
    ), f"Invalid characters in text (missing from language): {','.join([f'{c} ({unicodedata.name(c)})' for c in invalid_chars])}"

    # Generate clips
    if is_subtitle:
        clips, unlabelled_clips, clip_lengths = generate_clips_from_subtitles(
            audio_path,
            subs,
            transcription_model,
            output_path,
            logging,
            min_length,
            max_length,
            min_confidence,
        )
    else:
        clips, unlabelled_clips, clip_lengths = generate_clips_from_textfile(
            audio_path,
            text,
            transcription_model,
            output_path,
            logging,
            min_length,
            max_length,
            min_confidence,
        )

    assert clips, "No audio clips could be generated"

    if combine_clips:
        logging.info("Combining clips")
        clips, clip_lengths = clip_combiner(audio_path, output_path, clips, max_length)

    # Add silence
    silence = AudioSegment.silent(duration=int(silence_padding * 1000))
    for clip in clips:
        add_silence(os.path.join(output_path, clip["name"]), silence)

    # Move unused clips
    clip_names = [clip["name"] for clip in clips]
    for filename in os.listdir(output_path):
        if filename not in clip_names:
            if filename in unlabelled_clips:
                os.rename(os.path.join(output_path, filename), os.path.join(unlabelled_path, filename))
            else:
                os.remove(os.path.join(output_path, filename))

    # Produce alignment file
    logging.info(f"Produced {len(clips)} final clips")
    with open(forced_alignment_path, "w", encoding=CHARACTER_ENCODING) as result_file:
        result_file.write(json.dumps(clips, ensure_ascii=False, indent=4))

    # Produce metadata file
    with open(label_path, "w", encoding=CHARACTER_ENCODING) as f:
        for fragment in clips:
            f.write(f"{fragment['name']}|{fragment['text']}\n")
    logging.info("Generated clips")

    return clip_lengths


if __name__ == "__main__":
    """Generate clips"""
    parser = argparse.ArgumentParser(description="Split audio into snippets using forced align timings")
    parser.add_argument("-a", "--audio_path", help="Path to WAV file", type=str, required=True)
    parser.add_argument("-s", "--script_path", help="Path to text file", type=str, required=True)
    parser.add_argument(
        "-f", "--forced_alignment_path", help="Path to forced alignment JSON", type=str, default="align.json"
    )
    parser.add_argument("-o", "--output_path", help="Path to save snippets", type=str, default="wavs")
    parser.add_argument("-u", "--unlabelled_path", help="Path to save unlabelled clips", type=str, default="unlabelled")
    parser.add_argument(
        "-l", "--label_path", help="Path to save snippet labelling text file", type=str, default="metadata.csv"
    )
    parser.add_argument("--min_length", help="Minumum snippet length", type=float, default=1.0)
    parser.add_argument("--max_length", help="Maximum snippet length", type=float, default=10.0)
    parser.add_argument("--silence_padding", help="Silence padding on the end of the clip", type=int, default=0.1)
    parser.add_argument("--min_confidence", help="Minimum clip confidendence", type=float, default=0.85)
    args = parser.parse_args()

    clip_lengths = clip_generator(**vars(args))

    total_audio = sum(clip_lengths)
    total_clips = len(clip_lengths)
    minutes = int(total_audio / 60)
    seconds = total_audio - (minutes * 60)
    print(f"Total clips = {total_clips} (average of {total_audio/total_clips} seconds per clip)")
    print(f"Total audio = {minutes} minutes, {seconds} seconds")
    print(f"Audio saved to {args.output_path}. Text file save to {args.label_path}")

    import matplotlib.pyplot as plt

    plt.hist(clip_lengths, bins=10)
    plt.show()
