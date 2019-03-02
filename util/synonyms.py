import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict

model = Word2Vec.load('./model/word2vec/20190218.all.w2v')

def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, str):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


def _levenshtein_distance(sentence1, sentence2):
    '''
    Return the Levenshtein distance between two strings.
    Based on:
        http://rosettacode.org/wiki/Levenshtein_distance#Python
    '''
    first = any2utf8(sentence1).decode('utf-8', 'ignore')
    second = any2utf8(sentence2).decode('utf-8', 'ignore')
    sentence1_len, sentence2_len = len(first), len(second)
    maxlen = max(sentence1_len, sentence2_len)
    if sentence1_len > sentence2_len:
        first, second = second, first

    distances = range(len(first) + 1)
    for index2, char2 in enumerate(second):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(first):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                             distances[index1 + 1],
                                             new_distances[-1])))
        distances = new_distances
    levenshtein = distances[-1]
    d = float((maxlen - levenshtein)/maxlen)
    # smoothing
    s = (sigmoid(d * 6) - 0.5) * 2
    # print("smoothing[%s| %s]: %s -> %s" % (sentence1, sentence2, d, s))
    return s

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def find_synonyms(query_word, threshold=0.75):
    positives = defaultdict(int)
    positives[query_word] = 1
    while True:
        candidate = model.wv.most_similar(positive=list(positives.keys()), topn=50)
        size = len(positives)
        for key, value in candidate:
            sim = model.wv.similarity(query_word, key)
            if sim >= float(threshold) and max([_levenshtein_distance(key, y) for y in positives.keys()]) >= 0.9:
                positives[key] = sim
        if len(positives) == size:
            break
    return positives


def is_in_vocab(query_word):
    if query_word in model.wv.vocab:
        return True
    else:
        return False