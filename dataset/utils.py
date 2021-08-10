from difflib import SequenceMatcher


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
