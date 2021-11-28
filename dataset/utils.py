from difflib import SequenceMatcher
from string import punctuation, digits


def similarity(a, b):
    """
    Returns the similarity between two strings.

    Parameters
    ----------
    a : str
        String a
    b : str
        String b

    Returns
    -------
    float
        Similarity score (between 0-1)
    """
    return SequenceMatcher(None, a, b).ratio()


def add_suffix(filename, suffix):
    """
    Adds a suffix to a filename.

    Parameters
    ----------
    filename : str
        Current filename
    suffix : str
        String to add to the filename

    Returns
    -------
    str
        New filename
    """
    name_without_filetype = filename.split(".")[0]
    return filename.replace(name_without_filetype, name_without_filetype + "-" + suffix)


def get_invalid_characters(text, symbols):
    """
    Returns all invalid characters in text

    Parameters
    ----------
    text : str
        String to check
    symbols : list
        List of symbols that are valid

    Returns
    -------
    set
        All invalid characters
    """
    return set([c for c in text if c not in symbols and c not in punctuation and c not in digits])
