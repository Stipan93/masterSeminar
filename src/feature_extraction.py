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


def get_entity_type(entity):
    if entity == 'O':
        return entity
    else:
        return entity.split('-')[-1]


def get_next(words, i, offset):
    if i+offset < len(words):
        return words[i + offset]
    else:
        return None


def get_index_label(index):
    if index == 0:
        return 'B-'
    else:
        return 'I-'


def get_topic(word, words_on_i):
    if word is None:
        return 'unknown'
    topics = words_on_i.get(word.stem, None)
    if topics is not None:
            return topics[0]
    return 'unknown'
