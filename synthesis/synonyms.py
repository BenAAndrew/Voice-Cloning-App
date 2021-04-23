from dataset.transcribe import transcribe

import nltk

nltk.download("wordnet")
from nltk.corpus import wordnet


def get_synonyms(word):
    """
    Generates a list of synonyms for a word.

    Parameters
    ----------
    word : str
        Word to find synonyms of

    Returns
    -------
    list
        List of synonyms
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if "_" not in l.name():
                synonyms.add(l.name().lower())
    synonyms.discard(word)
    return list(synonyms)


def evalulate_audio(audio, text):
    """
    Gets list of words not recognised in the audio.
    Compares the transcription and given text.

    Parameters
    ----------
    audio : str
        Path to audio file
    text : str
        Synthesised text

    Returns
    -------
    set
        Set of words not recognised in the audio
    """
    results = transcribe(audio)
    original_words = text.split(" ")
    produced_words = results.split(" ")
    return set(original_words) - set(produced_words)


def get_alternative_word_suggestions(audio, text):
    """
    Produces a list of synonyms for each word not recognised in audio.
    This can be used to suggest word replacements to the user.

    Parameters
    ----------
    audio : str
        Path to audio file
    text : str
        Synthesised text

    Returns
    -------
    dict
        Words with a list of synonyms if the word was not recognised in the audio
    """
    all_words = text.split(" ")
    result = {word: [] for word in all_words}
    poor_words = evalulate_audio(audio, text)
    for word in poor_words:
        result[word] = get_synonyms(word)
    return result
