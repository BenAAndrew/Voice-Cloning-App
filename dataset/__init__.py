from string import punctuation, digits


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
