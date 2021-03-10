import argparse
from collections import OrderedDict
import re


if __name__ == "__main__":
    """ Script to clean Kindle book text file"""
    parser = argparse.ArgumentParser(description="Script to clean Kindle book text file")
    parser.add_argument("-i", "--input", help="Input path", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output text file path", type=str, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        book = f.readlines()

    # Remove duplicates
    book = list(OrderedDict.fromkeys(book))

    # Remove markdown elements and 'dirty' characters
    book = [re.sub("[^0-9a-zA-Z,.!?'\-\n ]+", "", line) for line in book if not line.startswith("![")]

    with open(args.output, "w") as f:
        for line in book:
            f.write(line)
            f.write("\n")
