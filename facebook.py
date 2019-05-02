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
import gc
from pymongo import MongoClient
from bson import json_util
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from math import exp

from util.synonyms import find_synonyms, find_candidate, is_in_vocab
from util.api import *

def signal_handler(signal, frame):
    sys.exit(0)

ALPHA = 0.25
PARAM_K1 = 1.2
PARAM_B = 0.75


FB_AVG_LENGTH = {
    'title': (32739) / 1404,
    'article': (39839103 + 46215754) / (1491207 + 1547773),
    'comment': (24586783) / 1547774
}
fb_max_likes_count = 177250
fb_max_commnets_count = 82453
fb_total_likes_count = 233203525
fb_total_comments_count = 24586783

def preload(post_tag=False):
    fb_posts_list = list()
    if post_tag ==True:
        options = default_options(3, 6)
        options['sentence'] = {'$ne': None}
        fields = default_fields()
        fields['sentence'] = 1
        fields['likes_count'] = 1
        fields['positive'] = 1
        fields['negative'] = 1
        # fields['reaction_count'] = 1
        # fields['attachments'] = 1
        cnt = 0
        for post in fb_posts.find(options, fields, no_cursor_timeout=True):
            cnt += 1
            sys.stdout.write(f'\r{cnt}')
            fb_posts_list.append(post)
        print()

    fb_objects_ref = defaultdict(dict)
    fields = {
        'id': 1,
        'type': 1,
        'name': 1,
        'words_count': 1,
        'like': 1,
        'tag': 1
    }
    for obj in fb_objects.find({}, fields):
        fb_objects_ref[obj['id']] = obj

    # return fb_objects_ref
    return fb_posts_list, fb_objects_ref

def BM25F(fields):
    global PARAM_K1
    global PARAM_B

    weight = 0
    for key, props in fields.items():
        weight += ( props['freq'] * props['boost'] ) / (1 - PARAM_B + PARAM_B * (props['dl'] / FB_AVG_LENGTH[key]))
    return weight / (PARAM_K1 + weight)

fb_posts_list, fb_objects_ref = preload(True)
print('preloading finished!')


def calculate(start, end, prog, transformation, relation_boosts):

    obj_score = defaultdict(lambda: defaultdict(int))
    history = defaultdict(dict)
    global_comments_count = list()
    global_likes_count = list()

    global fb_posts_list
    options = default_options(start, end)
    options['sentence'] = {'$ne': None}
    fields = default_fields()
    fields['sentence'] = 1
    fields['likes_count'] = 1
    fields['positive'] = 1
    fields['negative'] = 1
    print(options)
    cnt = 0
    start = strptime(start)
    end = strptime(end)
    # for post in fb_posts.find(options, fields, no_cursor_timeout=True).sort([('datetime_pub', 1)]):
    for post in fb_posts_list:
        if post['datetime_pub'] < start or post['datetime_pub'] >= end: continue
        if cnt % 100000 == 0:
            sys.stdout.write(f"\r{cnt}\n")
        cnt += 1

        _id = post['id']
        parent_id = post['parent_id']
        parent_name = post['parent_name']
        article_content = post['sentence']
        comments_count = post['comments_count']
        comments_content = post['comments_content']
        # reaction = {} if str(post['reaction_count']) == 'nan' else json.loads(post['reaction_count'])
        positive = post['positive']
        negative = post['negative']
        likes_count = post['likes_count']
        datetime_pub = post['datetime_pub']

        obj_score[parent_id]['id'] = parent_id
        obj_score[parent_id]['name'] = parent_name
        obj_score[parent_id]['total_comments_count'] += comments_count
        obj_score[parent_id]['total_posts_count'] += 1

        if ( transformation == 'sqrt' ):
            trans_comments_count = np.sqrt(comments_count)
            trans_likes_count = np.sqrt(likes_count)
        elif ( transformation == 'log' ):
            trans_comments_count = np.log1p(comments_count)
            trans_likes_count = np.log1p(likes_count)
        else:
            trans_comments_count = comments_count
            trans_likes_count = likes_count
        
        global_comments_count.append(trans_comments_count)
        global_likes_count.append(trans_likes_count)

        fields = defaultdict(lambda: defaultdict(int))
        if str(relation_boosts['title']) != '0':
            fields['title']['freq'] = len(prog.findall(parent_name))
            fields['title']['dl'] = post['parent_words_count']
            fields['title']['boost'] = float(relation_boosts['title'])
        if str(relation_boosts['article']) != '0':
            fields['article']['freq'] = len(prog.findall(article_content))
            fields['article']['dl'] = post['words_count']
            fields['article']['boost'] = float(relation_boosts['article'])
        if str(relation_boosts['comment']) != '0':
            fields['comment']['freq'] = len(prog.findall(comments_content))
            fields['comment']['dl'] = post['comments_words_count']
            fields['comment']['boost'] = float(relation_boosts['comment'])
        relation = BM25F(fields)

        if (( comments_count > 0  or likes_count > 0) and relation > 0):
            history[parent_id][_id] = {
                'relation': relation,
                'likes_count': likes_count,
                'comments_count': comments_count,
                'trans_comments_count': trans_comments_count,
                'trans_likes_count': trans_likes_count,
                'datetime_pub': datetime_pub,
                'positive': positive,
                'negative': negative,
                # 'type': '' if str(post['attachments']) == 'nan' else json.loads(post['attachments'])[0]['type'],
            }

    return obj_score, history, global_comments_count, global_likes_count

async def fb_query(websocket, path):
    global fb_posts_list
    global fb_objects_ref

    try:
        while True:
            req = json.loads(await websocket.recv())['req']
            it = time() 
            print(req)
            query_word = req['keywords']
            transformation = req['transformation']
            relation_boosts = req['relation_param']
            stats_type = req['stats_type']
            start = req['start']
            end = req['end']

            try: 
                wordset = query_word.split('|')
                prog = re.compile( "|".join(list(map(lambda x: "(" + x + ")", wordset))) )
            except:
                print('query_word error: ' + query_word)
                prog = re.compile("(Empty)")

            obj_score, history, global_comments_count, global_likes_count = calculate(start, end, prog, transformation, relation_boosts)

            global_comments_count = np.array(global_comments_count).astype(np.float32).reshape(-1, 1)
            global_likes_count = np.array(global_likes_count).astype(np.float32).reshape(-1, 1)
            
            comments_scaler = MinMaxScaler().fit(global_comments_count)
            likes_scaler = MinMaxScaler().fit(global_likes_count)    

            for parent_id, doc in history.items():
                for _id, props in doc.items():
                    relation = props['relation']
                    comments_count = props['comments_count']
                    likes_count = props['likes_count']
                    positive = props['positive']
                    negative = props['negative']
                    normalized_comments_count = comments_scaler.transform(np.array(props['trans_comments_count']).astype(np.float32).reshape(-1, 1))[0][0]
                    normalized_likes_count = likes_scaler.transform(np.array(props['trans_likes_count']).astype(np.float32).reshape(-1, 1))[0][0]

                    if (stats_type == 'likes'): score = relation * normalized_likes_count
                    elif (stats_type == 'comments'): score = relation * normalized_comments_count
                    else: score = relation * (normalized_comments_count + normalized_likes_count)
                    history[parent_id][_id]['score'] = float(score)
                    history[parent_id][_id]['normalized_comments_count'] = float(normalized_comments_count)

                    obj_score[parent_id]['relation'] += relation
                    obj_score[parent_id]['likes_count'] += likes_count
                    obj_score[parent_id]['comments_count'] += comments_count
                    obj_score[parent_id]['positive'] += positive
                    obj_score[parent_id]['negative'] += negative
                    obj_score[parent_id]['posts_count'] += 1
                    obj_score[parent_id]['score'] += float(score)
                    
            result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'] * (obj[1]['posts_count'] / obj[1]['total_posts_count']), reverse=True)
            data = list()
            tmp = defaultdict(dict)
            for obj, props in result:
                if props["score"] > 0:
                    tmp[obj] = history[obj]
                    row = {
                        'id': props['id'],
                        'name': props['name'],
                        'relation': '{0:.6f}'.format(props['relation'] / props['posts_count']),
                        'score': props['score'] * (props['posts_count'] / props['total_posts_count']),
                        'comments_count': props['comments_count'],
                        'likes_count': props['likes_count'],
                        'positive': 0 if (props['positive'] + props['negative']) == 0 else '{:.4f}'.format(props['positive'] / (props['positive'] + props['negative'])),
                        'negative': 0 if (props['positive'] + props['negative']) == 0 else '{:.4f}'.format(props['negative'] / (props['positive'] + props['negative'])),
                        'posts_count': props['posts_count'],
                        'total_comments_count': props['total_comments_count'],
                        'total_posts_count': '{0:.4f}'.format( props['posts_count'] / props['total_posts_count']),
                        'avg_comments_count': '{0:.2f}'.format(props['comments_count'] / props['posts_count']),
                        'like': -1 if 'like' not in fb_objects_ref[obj] else fb_objects_ref[obj]['like'],
                        'category': 'unknown' if 'type' not in fb_objects_ref[obj] else fb_objects_ref[obj]['type'],
                        'tag': "" if 'tag' not in fb_objects_ref[obj] else " ".join(fb_objects_ref[obj]['tag'])
                    }
                    data.append(row)
            ret = json.dumps({'data': data, 'history': tmp}, default=json_util.default)
            print(f'\n{time() - it}')
            gc.collect()
            await websocket.send(ret)
    except:
        await websocket.send('Error')
        signal.signal(signal.SIGINT, signal_handler)
        print(sys.exc_info()[0:2])
        print(traceback.extract_tb(sys.exc_info()[2]))
        sys.exit(0)

asyncio.get_event_loop().run_until_complete( websockets.serve(fb_query, 'localhost', 9526, close_timeout=15000) )
asyncio.get_event_loop().run_forever()