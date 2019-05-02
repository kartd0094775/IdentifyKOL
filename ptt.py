import asyncio
import websockets
import signal
import traceback

from collections import defaultdict
from time import sleep, time
import pandas as pd
import numpy as np
import pymongo
import json
import operator
import sys
import re
from pymongo import MongoClient
from bson import json_util
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from math import exp

from util.synonyms import find_synonyms, find_candidate, is_in_vocab
from util.api import *

PARAM_K1 = 2.0
PARAM_B = 0.75


PTT_AVG_LENGTH = {
    'title': 10838625 / 2632314,
    'article': 210163557 / 2632314,
    'comment': 180204594 / 2632314
}

def preload():
    options = {}
    options['content'] = {'$ne': None}
    fields = default_fields()
    fields['titles_words_count'] = 1
    fields['words_count'] = 1
    fields['comments_words_count'] = 1
    fields['comments_content'] = 1
    fields['reaction_count'] = 1
    ptt_posts_list = list(ptt_posts.find(options, fields, no_cursor_timeout=True).sort([('datetime_pub', 1)]))

    return ptt_posts_list

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def BM25F(fields):
    global PARAM_K1
    global PARAM_B

    weight = 0
    for key, props in fields.items():
        weight += ( props['freq'] * props['boost'] ) / (1 - PARAM_B + PARAM_B * (props['dl'] / PTT_AVG_LENGTH[key]))
    return weight / (PARAM_K1 + weight)

def calculate(start, end, prog, transformation, relation_boosts):

    obj_score = defaultdict(lambda: defaultdict(int))
    history = defaultdict(dict)
    global_comments_count = list()

    options = default_options(start, end)
    options['content'] = {'$ne': None}
    fields = default_fields()
    fields['uniID'] = 1
    fields['author'] = 1
    fields['title'] = 1
    fields['title_words_count'] = 1
    fields['content'] = 1
    fields['words_count'] = 1
    fields['comments_content'] = 1
    fields['comments_words_count'] = 1
    fields['reaction_count'] = 1
    fields['url'] = 1
 
    # for post in fb_posts_list:
        # if post['datetime_pub'] < strptime(start) or post['datetime_pub'] >= strptime(end): continue
    cnt = 0
    for post in ptt_posts.find(options, fields, no_cursor_timeout=True).sort([('datetime_pub', 1)]):
        if cnt % 10000 == 0:
            sys.stdout.write(f"\r{cnt}\n")
        cnt += 1

        _id = post['uniID']
        parent_id = post['author']
        title = "" if 'title' not in post else post['title']
        article_content = "" if 'content' not in post else post['content']
        comments_count = post['comments_count']
        comments_content = "" if 'comments_content' not in post else " ".join(post['comments_content'])
        reaction = {} if 'reaction_count' not in post else post['reaction_count']
        positive = 0 if len(reaction) == 0 else reaction['good']
        negative = 0 if len(reaction) == 0 else reaction['bad']
        url = "" if 'url' not in post else post['url']
        datetime_pub = post['datetime_pub']

        obj_score[parent_id]['id'] = parent_id
        obj_score[parent_id]['total_comments_count'] += comments_count
        obj_score[parent_id]['total_posts_count'] += 1

        if ( transformation == 'sqrt' ):
            trans_comments_count = np.sqrt(comments_count)
        elif ( transformation == 'log' ):
            trans_comments_count = np.log1p(comments_count)
        else:
            trans_comments_count = comments_count
        
        global_comments_count.append(trans_comments_count)

        fields = defaultdict(lambda: defaultdict(int))
        fields['title']['freq'] = len(prog.findall(title))
        fields['title']['dl'] = 0 if 'title_words_count' not in post else post['title_words_count']
        fields['title']['boost'] = float(relation_boosts['title'])
        fields['article']['freq'] = len(prog.findall(article_content))
        fields['article']['dl'] = 0 if 'words_count' not in post else post['words_count']
        fields['article']['boost'] = float(relation_boosts['article'])
        fields['comment']['freq'] = len(prog.findall(" ".join(comments_content)))
        fields['comment']['dl'] = 0 if 'comments_words_count' not in post else post['comments_words_count']
        fields['comment']['boost'] = float(relation_boosts['comment'])
        relation = BM25F(fields)

        if ( comments_count > 0 and relation > 0):
            history[parent_id][_id] = {
                'relation': relation,
                'comments_count': comments_count,
                'trans_comments_count': trans_comments_count,
                'datetime_pub': datetime_pub,
                'positive': positive,
                'negative': negative,
                'url': url
            }

    return obj_score, history, global_comments_count

async def ptt_query(websocket, path):

    try:
        while True:
            req = json.loads(await websocket.recv())['req']
            it = time() 
            print(req)
            query_word = req['keywords']
            transformation = req['transformation']
            relation_boosts = req['relation_boosts']
            start = req['start']
            end = req['end']

            wordset = query_word.split('|') 
            prog = re.compile( "|".join(list(map(lambda x: "(" + x + ")", wordset))) )

            obj_score, history, global_comments_count = calculate(start, end, prog, transformation, relation_boosts)

            global_comments_count = np.array(global_comments_count).astype(np.float32).reshape(-1, 1)
            comments_scaler = MinMaxScaler().fit(global_comments_count)  

            for parent_id, doc in history.items():
                for _id, props in doc.items():
                    comments_count = props['comments_count']
                    positive = props['positive']
                    negative = props['negative']
                    relation = props['relation']
                    normalized_comments_count = comments_scaler.transform(np.array(props['trans_comments_count']).astype(np.float32).reshape(-1, 1))[0][0]
                    score = relation * normalized_comments_count
                                    
                    history[parent_id][_id]['score'] = score
                    history[parent_id][_id]['normalized_comments_count'] = normalized_comments_count
                    
                    obj_score[parent_id]['score'] += score
                    obj_score[parent_id]['posts_count'] += 1
                    obj_score[parent_id]['positive'] += positive
                    obj_score[parent_id]['negative'] += negative
                    obj_score[parent_id]['comments_count'] += comments_count
                    obj_score[parent_id]['relation'] += relation
                    
            result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'] * (obj[1]['posts_count'] / obj[1]['total_posts_count']), reverse=True)
            data = list()
            tmp = defaultdict(dict)
            for obj, props in result:
                if props["score"] > 0:
                    tmp[obj] = history[obj]
                    row = {
                        'id': props['id'],
                        'relation': '{0:.6f}'.format(props['relation'] / props['posts_count']),
                        'score': props['score'] * (props['posts_count'] / props['total_posts_count']),
                        'comments_count': props['comments_count'],
                        'positives': props['positives'] / props['comments_count'],
                        'negative': props['negative'] / props['comments_count'],
                        'posts_count': props['posts_count'],
                        'total_comments_count': props['total_comments_count'],
                        'total_posts_count': '{0:.4f}'.format( props['posts_count'] / props['total_posts_count']),
                        'avg_comments_count': '{0:.2f}'.format(props['comments_count'] / props['posts_count']),
                    }
                    data.append(row)
            return data, history
            print(f'\n{time() - it}')
    except:
        await websocket.send('Error')
        signal.signal(signal.SIGINT, signal_handler)
        print(sys.exc_info()[0:2])
        print(traceback.extract_tb(sys.exc_info()[2]))
        sys.exit(0)

asyncio.get_event_loop().run_until_complete( websockets.serve(ptt_query, 'localhost', 9525, close_timeout=15000) )
asyncio.get_event_loop().run_forever()