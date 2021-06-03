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


def get_alternative_word_suggestions(text):
    """
    Produces a list of synonyms for each word.
    This can be used to suggest word replacements to the user.

    Parameters
    ----------
    text : str
        Synthesised text

    Returns
    -------
    dict
        Words with a list of synonyms
    """
    all_words = text.split(" ")
    return {word: get_synonyms(word) for word in all_words}
