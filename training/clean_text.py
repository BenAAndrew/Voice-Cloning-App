import argparse
import re

import inflect
from training import DEFAULT_ALPHABET

INFLECT_ENGINE = inflect.engine()
COMMA_NUMBER_RE = re.compile(r"([0-9][0-9\,]+[0-9])")
DECIMAL_NUMBER_RE = re.compile(r"([0-9]+\.[0-9]+)")
NUMBER_RE = re.compile(r"[0-9]+")
ORDINALS = re.compile(r"([0-9]+[st|nd|rd|th]+)")
CURRENCY = re.compile(r"([£|$|€]+[0-9]+)")
WHITESPACE_RE = re.compile(r"\s+")
ALLOWED_CHARACTERS_RE = re.compile("[^a-z ,.!?'-]+")
MONETARY_REPLACEMENT = {"$": " dollars", "£": " pounds", "€": " euros"}
ABBREVIATION_REPLACEMENT = {
    "mr.": "mister",
    "mrs.": "misess",
    "dr.": "doctor",
    "no.": "number",
    "st.": "saint",
    "co.": "company",
    "jr.": "junior",
    "maj.": "major",
    "gen.": "general",
    "drs.": "doctors",
    "rev.": "reverend",
    "lt.": "lieutenant",
    "hon.": "honorable",
    "sgt.": "sergeant",
    "capt.": "captain",
    "esq.": "esquire",
    "ltd.": "limited",
    "col.": "colonel",
    "ft.": "fort",
}


def clean_text(text, symbols=DEFAULT_ALPHABET, remove_invalid_characters=True):
    """
    Cleans text. This includes:
    - Replacing monetary terms (i.e. $ -> dollars)
    - Converting ordinals to full words (i.e. 1st -> first)
    - Converting numbers to their full word format (i.e. 100 -> one hundred)
    - Replacing abbreviations (i.e. dr. -> doctor)
    - Removing invalid characters (non utf-8 or invalid punctuation)

    Parameters
    ----------
    text : str
        Text to clean
    symbols : list (optional)
        List of valid symbols in text (default is English alphabet & punctuation)
    remove_invalid_characters : bool (optional)
        Whether to remove characters not in symbols list (default is True)

    Returns
    -------
    str
        Cleaned text
    """
    text = text.strip()
    text = text.lower()
    # Convert currency to words
    money = re.findall(CURRENCY, text)
    for amount in money:
        for key, value in MONETARY_REPLACEMENT.items():
            if key in amount:
                text = text.replace(amount, amount[1:] + value)
    # Convert ordinals to words
    ordinals = re.findall(ORDINALS, text)
    for ordinal in ordinals:
        text = text.replace(ordinal, INFLECT_ENGINE.number_to_words(ordinal))
    # Convert comma & decimal numbers to words
    numbers = re.findall(COMMA_NUMBER_RE, text) + re.findall(DECIMAL_NUMBER_RE, text)
    for number in numbers:
        text = text.replace(number, INFLECT_ENGINE.number_to_words(number))
    # Convert standard numbers to words
    numbers = re.findall(NUMBER_RE, text)
    for number in numbers:
        text = text.replace(number, INFLECT_ENGINE.number_to_words(number))
    # Replace abbreviations
    for key, value in ABBREVIATION_REPLACEMENT.items():
        text = text.replace(" " + key + " ", " " + value + " ")
    # Collapse whitespace
    text = re.sub(WHITESPACE_RE, " ", text)
    # Remove banned characters
    if remove_invalid_characters:
        text = "".join([c for c in text if c in symbols])
    return text


if __name__ == "__main__":
    """Script to clean text for training"""
    parser = argparse.ArgumentParser(description="Clean & improve text for training")
    parser.add_argument("-f", "--file", help="Text file path", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output text file path", type=str, required=True)
    args = parser.parse_args()

    with open(args.file) as f:
        rows = f.readlines()

    cleaned_text = []

    for row in rows:
        filename, text = row.split("|")
        text = clean_text(text)
        cleaned_text.append(f"{filename}|{text}")

    with open(args.output, "w") as f:
        for line in cleaned_text:
            f.write(line)
            f.write("\n")
