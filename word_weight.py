import numpy as np
import pickle as pkl
import sys
from util.api import *

with open('2018_fb_document_freq.pkl', 'rb') as f:
    df = pkl.load(f)
N = np.sum(list(df.values()))
avdl = (39839103 + 46215754) / (1491207 + 1547773)


def BM25(doc, dl):
    global N
    global df
    global avdl
    PARAM_K1 = 2.0
    PARAM_B = 0.75
    K = PARAM_K1 * (1 - PARAM_B + PARAM_B * (dl / avdl))    

    normalized = defaultdict(float)
    raw = list()
    for word, value in doc.items():
        idf = np.log((N - df[word] + 0.5) / (df[word] + 0.5))
        first = value * (PARAM_K1 + 1) / (value + K)
        raw.append(idf * first)
    raw = raw / np.linalg.norm(raw, 2, axis=0)
    
    for word, value in zip(doc.keys(), raw):
        normalized[word] = float(value)
    return normalized
        
    
def TFIDF(doc):
    global N
    global df
    
    normalized = defaultdict(float)
    raw = list()
    for word, value in doc.items():
        idf = np.log((1 + N) / (1 + df[word])) + 1
        raw.append(value * idf)
    raw = raw / np.linalg.norm(raw, 2, axis=0)
    
    for word, value in zip(doc.keys(), raw):
        normalized[word] = float(value)
    return normalized

if __name__ == "__main__":
    cnt = 0
    for post in fb_posts.find({'keywords': {'$ne': None}}, {'keywords': 1, 'words_count': 1}, no_cursor_timeout=True):
        sys.stdout.write(f'\r{cnt}')
        cnt += 1
        doc = post['keywords']
        bm25 = BM25(doc, post['words_count'])
        tfidf = TFIDF(doc)
        fb_posts.update_one({'_id': post['_id']}, {'$set': {'bm25': bm25, 'tfidf': tfidf}}, upsert=False)