"""
Credit: https://github.com/snakers4/silero-models

                    GNU AFFERO GENERAL PUBLIC LICENSE
                       Version 3, 19 November 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""

import torch
import warnings
from typing import List
from itertools import groupby


class Decoder:
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.blank_idx = self.labels.index("_")
        self.space_idx = self.labels.index(" ")

    def process(self, probs, wav_len, word_align):
        assert len(self.labels) == probs.shape[1]
        for_string = []
        argm = torch.argmax(probs, axis=1)
        align_list = [[]]
        for j, i in enumerate(argm):
            if i == self.labels.index("2"):
                try:
                    prev = for_string[-1]
                    for_string.append("$")
                    for_string.append(prev)
                    align_list[-1].append(j)
                    continue
                except:
                    for_string.append(" ")
                    warnings.warn('Token "2" detected a the beginning of sentence, omitting')
                    align_list.append([])
                    continue
            if i != self.blank_idx:
                for_string.append(self.labels[i])
                if i == self.space_idx:
                    align_list.append([])
                else:
                    align_list[-1].append(j)

        string = "".join([x[0] for x in groupby(for_string)]).replace("$", "").strip()

        align_list = list(filter(lambda x: x, align_list))

        if align_list and wav_len and word_align:
            align_dicts = []
            linear_align_coeff = wav_len / len(argm)
            to_move = min(align_list[0][0], 1.5)
            for i, align_word in enumerate(align_list):
                if len(align_word) == 1:
                    align_word.append(align_word[0])
                align_word[0] = align_word[0] - to_move
                if i == (len(align_list) - 1):
                    to_move = min(1.5, len(argm) - i)
                    align_word[-1] = align_word[-1] + to_move
                else:
                    to_move = min(1.5, (align_list[i + 1][0] - align_word[-1]) / 2)
                    align_word[-1] = align_word[-1] + to_move

            for word, timing in zip(string.split(), align_list):
                align_dicts.append(
                    {
                        "word": word,
                        "start_ts": round(timing[0] * linear_align_coeff, 2),
                        "end_ts": round(timing[-1] * linear_align_coeff, 2),
                    }
                )

            return string, align_dicts
        return string

    def __call__(self, probs: torch.Tensor, wav_len: float = 0, word_align: bool = False):
        return self.process(probs, wav_len, word_align)


def init_jit_model(model_path: str, device: torch.device = torch.device("cpu")):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, Decoder(model.labels)
