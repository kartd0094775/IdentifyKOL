from collections import defaultdict
from time import sleep, time
from os import walk
import pandas as pd
import numpy as np
import json
import operator
import sys
import pymongo
from pymongo import MongoClient
from collections import defaultdict
from time import mktime, time
from datetime import datetime

client = MongoClient('localhost', 27017)
database = client["research"]

fb_objects = database['2018_fb_objects']
fb_posts = database["2018_fb_posts"]
fb_comments = database['2018_fb_comments']

ptt_objects = database['2018_ptt_objects']
ptt_posts = database["2018_ptt_posts"]
ptt_comments = database['2018_ptt_comments']

def default_options(start, end):
    return {'datetime_pub': {'$gte': datetime(2018, start, 1), '$lt': datetime(2018, end, 1)}}

def default_fields():
    return {'id':1, 'parent_id':1, 'parent_name': 1, 'comments_count': 1, 'global_tfidf': 1, "datetime_pub": 1}

def rank_fb_posts(query_word, fb_id, start, end):
    post_score = defaultdict(list)

    options = default_options(start, end)
    options['parent_id'] = str(fb_id)
    field = default_field()
    for post in fb_posts.find(options, field, no_cursor_timeout=True):
        properties = [0, 0, 0, ''] # relation, comments_count, words
        if 'global_tfidf' in post:
            _id = post['id']
            post_tfidf = post['global_tfidf']
            comments_count = post['comments_count']
            cnt, score, words = 0, 0, list()
            for word, value in post_tfidf:
                if len(word) == 1 and ord(word) == 65039: continue # if word == ''
                if word in model.wv:
                    words.append(word)
                    score += model.wv.similarity(query_word, word) * value
                    cnt += 1
                if cnt == 7: break
            if cnt > 0:
                words = ' '.join(words)
                properties[0] = score * np.log(1 + comments_count) / cnt
                properties[1] = score / cnt
                properties[2] = comments_count
                properties[3] = words
                post_score[_id] = properties
    result = sorted(post_score.items(), key=lambda obj: obj[1][0], reverse=True)

    data = defaultdict(list)
    for idx, value in result:
        data['id'].append(idx)
        data['score'].append(value[0])
        data['relation'].append(value[1])
        data['comments_count'].append(value[2])
        data['words'].append(value[3])
    df = pd.DataFrame(data)
    return df

def to_dataframe(result):
    data = defaultdict(list)
    for obj, value in result:
        obj = find_fb_obj(obj)
        data['id'].append(obj['id'])
        data['name'].append(obj['name'])
        data['relation'].append(value[0])
        data['comments_count'].append(value[1])
        data['posts_count'].append(value[2])
        data['score'].append(value[0] / np.log(1+value[2]))
    df = pd.DataFrame(data)
    return df