import re
import numpy as np

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


def make_feature_vec(word, prev, prev_prev, next, next_next):
    vec = []
    vec.append(get_entity(prev_prev))
    vec.append(get_entity(prev))

    vec.append(alpha_numeric(word.token))
    vec.append(all_digits(word.token))
    vec.append(all_capitalized(word.token))

    # extract 3gram chars from and group the by entity

    # vec.append(get_stem(prev_prev))
    # vec.append(get_stem(prev))
    # vec.append(get_stem(word))
    # vec.append(get_stem(next))
    # vec.append(get_stem(next_next))

    vec.append(is_capitalized(get_token(prev_prev)))
    vec.append(is_capitalized(get_token(prev)))
    vec.append(is_capitalized(word.token))
    vec.append(is_capitalized(get_token(next)))
    vec.append(is_capitalized(get_token(next_next)))

    return vec


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


def get_features(data):
    features = []
    Y = []
    for doc in data.documents:
        for sentance in doc.sentences:
            n = len(sentance.words)
            for i in range(n):
                prev = get_prev(sentance.words, i, 1)
                prev_prev = get_prev(sentance.words, i, 2)
                next = get_next(sentance.words, i, 1)
                next_next = get_next(sentance.words, i, 2)
                features.append(make_feature_vec(sentance.words[i], prev, prev_prev, next, next_next))
                Y.append(sentance.words[i].entity)
    return np.array(features), np.array(Y)
