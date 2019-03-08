import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import re
import jieba
import subprocess 
from gensim.test.utils import get_tmpfile, common_texts
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontManager 
from pylab import mpl 


jieba.load_userdict('C:/Users/waynet/venv/Lib/site-packages/jieba/dict.blue.txt')

def load_stopwords():
    with open('util/stopwords.pkl', 'rb') as f:
         stopwords = pkl.load(f)
    return stopwords

def load_symbols():
    ret = []
    with open('util/symbols_20181216.txt', 'r', encoding='utf-8') as f:
        rows = f.readlines()
    f.close()
    for row in rows:
        if row[:-1] not in ret:
            ret.append(row[:-1])
    return ret

def load_pattern():
    symbols = load_symbols()
    symbols += ['\n', '\r\n', '\r']
    symbols_str = ''
    for symbol in symbols:
        if symbol in '[]()-': symbol = '\\' + symbol
        symbols_str += symbol
    return re.compile(r'([0-9]+|\.+|[a-zA-Z])|[{}]+'.format(symbols_str))

def tokenize(corpus, stopwords=load_stopwords(), pattern=re.compile(r'[\WA-Za-z0-9]+'), length_constraint=2):
    tokenized_corpus = []
    for doc in corpus:
        tokenized_doc = jieba.lcut(doc)
        words = []
        for word in tokenized_doc:
            if word in stopwords or pattern.match(word): continue
            elif len(word) < length_constraint: continue
            else: words.append(word)
        tokenized_corpus.append(words)
    return tokenized_corpus