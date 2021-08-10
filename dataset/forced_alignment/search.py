from dataset.utils import similarity


def ngrams(s, size):
    """
    Credit: https://github.com/mozilla/DSAlign

    Lists all appearances of all N-grams of a string from left to right.
    :param s: String to decompose
    :param size: N-gram size
    :return: Produces strings representing all N-grams
    """
    window = len(s) - size
    if window < 1 or size < 1:
        if window == 0:
            yield s
        return
    for i in range(0, window + 1):
        yield s[i : i + size]


class FuzzySearch(object):
    """
    Credit: https://github.com/mozilla/DSAlign

    FuzzySearch class to search text file.

    Parameters
    ----------
    text : str
        Text from text file
    Other optional parameters
    """

    def __init__(
        self,
        text,
        max_candidates=10,
        candidate_threshold=0.92,
        match_score=100,
        mismatch_score=-100,
        gap_score=-100,
        char_similarities=None,
    ):
        self.text = text
        self.max_candidates = max_candidates
        self.candidate_threshold = candidate_threshold
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score
        self.char_similarities = char_similarities
        self.ngrams = {}
        for i, ngram in enumerate(ngrams(" " + text + " ", 3)):
            if ngram in self.ngrams:
                ngram_bucket = self.ngrams[ngram]
            else:
                ngram_bucket = self.ngrams[ngram] = []
            ngram_bucket.append(i)

    def sim_align(self, a, start, end):
        source = self.text[start:end]
        words = source.split(" ")
        best = ""
        best_score = 0
        for i in range(len(words)):
            for j in range(i, len(words)):
                t = " ".join(words[i:j])
                score = similarity(a, t)
                if score > best_score:
                    best = t
                    best_score = score

        start = self.text.index(best)
        end = start + len(best)
        return start, end, best_score

    def find_best(self, look_for, start=0, end=-1):
        end = len(self.text) if end < 0 else end
        if end - start < 2 * len(look_for):
            return self.sim_align(look_for, start, end)
        window_size = len(look_for)
        windows = {}
        for i, ngram in enumerate(ngrams(" " + look_for + " ", 3)):
            if ngram in self.ngrams:
                ngram_bucket = self.ngrams[ngram]
                for occurrence in ngram_bucket:
                    if occurrence < start or occurrence > end:
                        continue
                    window = occurrence // window_size
                    windows[window] = (windows[window] + 1) if window in windows else 1
        candidate_windows = sorted(windows.keys(), key=lambda w: windows[w], reverse=True)
        best = (-1, -1, 0)
        last_window_grams = 0.1
        for window in candidate_windows[: self.max_candidates]:
            ngram_factor = windows[window] / last_window_grams
            if ngram_factor < self.candidate_threshold:
                break
            last_window_grams = windows[window]
            interval_start = max(start, int((window - 1) * window_size))
            interval_end = min(end, int((window + 2) * window_size))
            search_result = self.sim_align(look_for, interval_start, interval_end)
            if search_result[2] > best[2]:
                best = search_result
        return best
