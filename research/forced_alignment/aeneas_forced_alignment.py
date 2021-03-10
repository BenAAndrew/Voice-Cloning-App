import argparse
import logging
import json

logging.getLogger().setLevel(logging.INFO)

from aeneas.executetask import ExecuteTask
from aeneas.task import Task


def force_align(audio_path, text_path, output_path, min_length=1.0, max_length=10.0, logging=logging):
    sentences = []
    task = Task(config_string=u"task_language=eng|is_text_type=plain|os_task_file_format=json")
    task.audio_file_path_absolute = audio_path
    task.text_file_path_absolute = text_path
    task.sync_map_file_path_absolute = output_path
    logging.info("Aligning audio and text...")
    ExecuteTask(task).execute()
    logging.info("Aligned audio and text")

    for fragment in task.sync_map_leaves():
        if fragment.length > min_length and fragment.length < max_length and fragment.text:
            sentences.append(
                {
                    "start": float(fragment.begin),
                    "end": float(fragment.end),
                    "length": float(fragment.length),
                    "text": fragment.text,
                }
            )

    with open(output_path, "w") as f:
        json.dump(sentences, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Force align audio to text")
    parser.add_argument("-a", "--audio_path", help="Path to audio file", type=str, required=True)
    parser.add_argument("-t", "--text_path", help="Path to text file", type=str, required=True)
    parser.add_argument("-o", "--output_path", help="Path to output file", type=str, required=True)
    parser.add_argument("-m", "--min_length", help="Minimum clip length", type=float, default=1.0)
    parser.add_argument("-x", "--max_length", help="Maximum clip length", type=float, default=10.0)
    args = parser.parse_args()

    import time

    start = time.time()
    force_align(**vars(args))

    print("DURATION", time.time() - start)
