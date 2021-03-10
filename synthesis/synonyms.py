from dataset.transcribe import transcribe

import nltk

nltk.download("wordnet")
from nltk.corpus import wordnet


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if "_" not in l.name():
                synonyms.add(l.name().lower())
    synonyms.discard(word)
    return list(synonyms)


def evalulate_audio(audio, text):
    results = transcribe(audio)
    original_words = text.split(" ")
    produced_words = results.split(" ")
    return set(original_words) - set(produced_words)


def get_alternative_word_suggestions(audio, text):
    all_words = text.split(" ")
    result = {word: [] for word in all_words}
    poor_words = evalulate_audio(audio, text)
    for word in poor_words:
        result[word] = get_synonyms(word)
    return result
