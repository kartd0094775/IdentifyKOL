import pandas as pd
import numpy as np
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


jieba.load_userdict('C:/Users/Waynet/venv/Lib/site-packages/jieba/dict.blue.txt')

def load_stopwords():
    stopwords = []
    with open('stopwords.txt', 'r') as f:
         words = f.readlines()
    f.close()
    for word in words:
        if word[:-1] not in stopwords:
            stopwords.append(word[:-1])
    return stopwords
def load_symbols():
    ret = []
    with open('symbols_20181216.txt', 'r', encoding='utf-8') as f:
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
    return re.compile('([0-9]+|\.+|[a-zA-Z])|[{}]+'.format(symbols_str))
def filter_empty_articles(articles):
    ret = []
    for article in articles:
        if type(article) is not float:
            ret.append(article)
    return ret
def to_sentence(document):
    ret = list()
    rule = re.compile('[\W]+')
    result = rule.split(document)
    for sentence in result:
        if len(sentence) > 0:
            ret.append(sentence)
    return ret

def tokenize(corpus, stopwords=load_stopwords(), pattern=re.compile('[\WA-Za-z0-9]+')):
    tokenized_corpus = []
    for doc in corpus:
        tokenized_doc = jieba.lcut(doc)
        words = []
        for word in tokenized_doc:
            if word in stopwords or pattern.match(word): continue
            words.append(word)
        tokenized_corpus.append(words)
    return tokenized_corpus

def train_word_embedding(filename, model):
    df = pd.read_csv('./data/fb_posts/{}'.format(filename), sep=',', encoding='utf-8')
    contents = filter_empty_articles(df['content'])
    tokenized_contents = tokenize(contents)
    model.train(tokenized_contents, total_examples=len(tokenized_contents), epochs=5)
    try:
        model.save('model/word2vec/{}.model'.format(filename[:-4]))
    except:
        print('cannnot save the model')
    return model