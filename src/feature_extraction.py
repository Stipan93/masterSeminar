import re

capitalized = "^[A-Z].*$"
allcapitalized = "^[A-Z]*$"
alldigits = "^[0-9]*$"
alphanumeric = "^[A-Za-z0-9]*$"


def get_entity(word):
    if word is None:
        return 'unknown'
    else:
        return word.entity


def get_token(word):
    if word is None:
        return 'unknown'
    else:
        return word.token


def get_stem(word):
    if word is None:
        return 'unknown'
    else:
        return word.stem


def is_capitalized(token):
    return re.search(capitalized, token) is not None


def all_capitalized(token):
    return re.search(allcapitalized, token) is not None


def alpha_numeric(token):
    return re.search(alphanumeric, token) is not None


def all_digits(token):
    return re.search(alldigits, token) is not None


def get_prev(words, i, offset):
    if i >= offset:
        return words[i-offset]
    else:
        return None


def get_next(words, i, offset):
    if i+offset < len(words):
        return words[i + offset]
    else:
        return None
