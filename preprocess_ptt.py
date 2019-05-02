import sys
import pymysql
import pymongo
import re
import itertools
import pickle as pkl
import pandas as pd
from pymongo import MongoClient
from collections import defaultdict

from util.preprocessing import *

client = MongoClient('localhost', 27017)
database = client['research']
ptt_posts = database['2018_ptt_posts']
ptt_objects = database['2018_ptt_objects']
ptt_comments = database['2018_ptt_comments']

if __name__ == "__main__":
    total_document_len = 0
    total_document_count = 0
    document_freq = defaultdict(int)

    cnt = 0
    for post in ptt_posts.find(no_cursor_timeout=True):
        sys.stdout.write(f'\r{cnt}')
        cnt += 1

        post['comments'] = list(ptt_comments.find({'parentID': post['uniID']}, no_cursor_timeout=True))
        post['comments_count'] = len(post['comments'])
        
        content = post['content']
        if str(content) != 'nan' and content != None:

            sentences = to_sentence(content)
            tokenized_content = tokenize(sentences, load_stopwords(), re.compile('[\Wa-zA-Z0-9]+'))
            post['sentence'] = " ".join(sentences)
            post['tokenized_content'] = tokenized_content
            post['keywords'] = defaultdict(int)
            for term in itertools.chain.from_iterable(tokenized_content):
                document_freq[term] += 1
                post['keywords'][term] += 1
            post['words_count'] = len(post['keywords'])

            total_document_len += post['words_count']
            total_document_count += 1

            ptt_posts.update_one({'_id': post['_id']}, {'$set': post}, upsert=False)

    with open('document_freq.pkl', 'wb') as f:
        pkl.dump(document_freq, f)

    with open('ptt_log.txt', 'w') as f:
        f.write(f'total_document_length: {total_document_len}\n')
        f.write(f'total_document_count: {total_document_count}\n')
        f.write(f'avg_document_length: {total_document_len / total_document_count}\n')
    
    