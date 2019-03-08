from collections import defaultdict
from flask import Blueprint
from flask import request
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
from sklearn.feature_extraction.text import TfidfVectorizer
from math import exp

from util.synonyms import find_synonyms, is_in_vocab
from util.api import *


query_page = Blueprint('query', 'query')

ALPHA = 0.25
PARAM_K1 = 2.0
PARAM_B = 0.75
avdl = (39839103 + 46215754) / (1491207 + 1547773)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def bm25(value, dl): 
    global PARAM_K1
    global PARAM_B
    global avdl

    K = PARAM_K1 * (1 - PARAM_B + PARAM_B * (dl / avdl))    
    return value * (PARAM_K1 + 1) / (value + K)

### preload ###
# fileds = {
#   'id': 1,
#   '
# }
# fb_posts_list = list(fb_posts.find(default_options(3, 6), fields, no_cursor_timeout=True).sort([('datetime_pub', 1)]))
fb_objects_ref = defaultdict(dict)
fields = {
    'id': 1,
    'type': 1,
    'name': 1,
    'like': 1,
    'tag': 1
}
for obj in fb_objects.find({}, fields):
    fb_objects_ref[obj['id']] = obj


@query_page.route('/fb/', methods=['POST'])
def fb_query():
    """
    Get relative fb_objects by keyword.
    ---
    tags:
        - query
    parameters:
        - name: payload
          in: body
          type: object
          properties:
              keyword:
                  type: string
              default: {"keyword":"化妝品", "start_date": "2018-01-01", "end_date": "2018-12-31"}
    responses:
        200:
            ret: return value
    """
    it = time()
    req = request.get_json()
    query_word = req['keyword']
    start = req['start']
    end = req['end']
    threshold = req['threshold']
    
    title_count = 0
    for obj in fb_objects_ref.values():
        if obj['name'] != None and query_word in obj['name']:
            title_count += 1
    title_count = float(np.log10( 9 + title_count))
    
    if is_in_vocab(query_word):
        wordset = find_synonyms(query_word, threshold)
        print(f'mode: Vocab, wordset: {wordset}')
        print(f'range: {start} - {end}')
    else:
        wordset = {query_word: 1}
        print(f'mode: Fulltext, wordset: {query_word}')
        print(f'range: {start} - {end}')

    obj_score = defaultdict(lambda: defaultdict(int))
    history = defaultdict(dict)

    options = default_options(start, end)
    fields = default_fields()
    fields['words_count'] = 1
    fields['reaction_count'] = 1
    fields['attachments'] = 1
    fields['sentence'] = 1

    cnt = 0
    for post in fb_posts.find(options, fields, no_cursor_timeout=True).sort([('datetime_pub', 1)]):
        if post['datetime_pub'] < datetime(2018, int(start), 1) or post['datetime_pub'] >= datetime(2018, int(end), 1): continue
        if cnt % 10000 == 0:
            sys.stdout.write(f"\r{cnt}")
        cnt += 1

        _id = post['id']
        parent_id = post['parent_id']
        parent_name = post['parent_name'] if post['parent_name'] != None else ''
        comments_count = post['comments_count']
        reaction_count = None if str(post['reaction_count']) == 'nan' else json.loads(post['reaction_count'])
        likes_count = 0 if reaction_count == None else int(np.sum(list(reaction_count.values())))
        datetime_pub = post['datetime_pub']

        obj_score[parent_id]['id'] = parent_id
        obj_score[parent_id]['name'] = parent_name
        obj_score[parent_id]['total_comments_count'] += comments_count
        obj_score[parent_id]['total_posts_count'] += 1
        title_relation = 0 if query_word not in parent_name else (3 / title_count)
        ALPHA = 0
        # if fulltext_mode == True and 'sentence' in post:
        #     freq = len(re.findall(f'{(query_word)}', post['sentence']))
        #     relation = bm25(freq, post['words_count'])
        # elif fulltext_mode == False and 'bm25' in post:
            # weight = lambda x: 0 if x not in post['bm25'] else post['bm25'][x]
            # relation = np.sqrt(np.sum([pow(weight(x), 2) for x in wordset.keys()]))
        if 'sentence' in post: 
            prog = re.compile( "|".join(list(map(lambda x: "(" + x + ")", list(wordset.keys())))) )
            freq = len(prog.findall(post['sentence']))
            relation = bm25(freq, post['words_count'])
            if relation > 0:
                relation = ALPHA * title_relation +  (1 - ALPHA) * relation
                normalized_comments_count = np.log(1 + comments_count)
                normalized_likes_count = np.log(1 + likes_count)
                # score = relation * (normalized_comments_count + normalized_likes_count)
                # score = relation * (np.sqrt(comments_count * 2.149)  + np.sqrt(likes_count))
                score = relation * (comments_count * 2.149 + likes_count)
                if score > 0:
                    history[parent_id][_id] = {
                        'relation': relation,
                        'likes_count': likes_count,
                        'comments_count': comments_count,
                        'normalized_comments_count': normalized_comments_count,
                        'datetime_pub': datetime_pub,
                        'score': score,
                        'type': '' if str(post['attachments']) == 'nan' else json.loads(post['attachments'])[0]['type'],
                    }
                obj_score[parent_id]['relation'] += relation
                obj_score[parent_id]['likes_count'] += likes_count
                obj_score[parent_id]['comments_count'] += comments_count
                obj_score[parent_id]['posts_count'] += 1                    
                obj_score[parent_id]['score'] += score

    # result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'] / (1 + np.log(1 + obj[1]['posts_count'])) * obj[1]['posts_count'] / obj[1]['total_posts_count'], reverse=True)
    # result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'] * (sigmoid(obj[1]['posts_count'] / obj[1]['total_posts_count'] * 6) - 0.5) * 2 , reverse=True)
    result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'] * obj[1]['posts_count'] / obj[1]['total_posts_count'], reverse=True)
    data = list()
    tmp = defaultdict(dict)
    for obj, props in result:
        if props["score"] > 0:
            tmp[obj] = history[obj]
            row = {
                'id': props['id'],
                'name': props['name'],
                'relation': '{0:.6f}'.format(props['relation'] / props['posts_count']),
                'score': props['score'] / (1 + np.log(1 + props['posts_count'])) * (props['posts_count'] / props['total_posts_count']),
                'comments_count': props['comments_count'],
                'likes_count': props['likes_count'],
                'posts_count': props['posts_count'],
                'total_comments_count': props['total_comments_count'],
                'total_posts_count': props['total_posts_count'],
                'ratio': (sigmoid(props['posts_count'] / props['total_posts_count'] * 6) - 0.5) * 2,
                'avg_comments_count': '{0:.2f}'.format(props['comments_count'] / props['posts_count']),
                'like': -1 if 'like' not in fb_objects_ref[obj] else fb_objects_ref[obj]['like'],
                'category': 'unknown' if 'type' not in fb_objects_ref[obj] else fb_objects_ref[obj]['type'],
                'tag': "" if 'tag' not in fb_objects_ref[obj] else " ".join(fb_objects_ref[obj]['tag'])
            }
            data.append(row)
    ret = json.dumps({'data': data, 'history': tmp}, default=json_util.default)
    print(f'\n{time() - it}')
    return ret

# fb/reaction
@query_page.route('/fb/reaction/', methods=['POST'])
def fb_reaction_query():
    """
    Get relative fb_objects by keyword.
    ---
    tags:
        - query
    parameters:
        - name: payload
          in: body
          type: object
          properties:
              keyword:
                  type: string
              default: {"keyword":"化妝品", "start_date": "2018-01-01", "end_date": "2018-12-31"}
    responses:
        200:
            ret: return value
    """
    pt = time()
    req = request.get_json()
    query_word = req['keyword']
    start = req['start']
    end = req['end']
    print(req)

    obj_score = dict()
    history = defaultdict(dict)
    alpha = 0.5
    
    for obj in fb_objects.find({'tfidf': {"$ne": None}}):
        tfidf = dict(obj['tfidf'])
        name = obj['name']
        title_relation = 0 if query_word not in name else 1

        if query_word in tfidf:
            properties = defaultdict(int)
            properties['relation'] = tfidf[query_word]

            options = default_options(start, end)
            options['parent_id'] = str(obj['id'])
            options['reaction_count'] = {'$ne': None, '$ne': float('nan')}
            field = default_fields()
            field['reaction_count'] = 1

            for post in fb_posts.find(options, fields, no_cursor_timeout=True).sort([('datetime_pub', 1)]):
                _id = post['id']
                parent_id = post['parent_id']
                comments_count = post['comments_count']
                datetime_pub = post['datetime_pub']
                reaction_count = json.loads(post['reaction_count'])
                reaction_count =int(np.sum(list(reaction_count.values()), axis=0))
                
                properties['total_reaction_count'] += reaction_count
                properties['total_comments_count'] += comments_count
                properties['total_posts_count'] += 1
                if 'global_tfidf' in post and query_word in dict(post['global_tfidf']):
                    relation = dict(post['global_tfidf'])[query_word]
                    normalized_reaction_count = np.log(1 + reaction_count)
                    normalized_comments_count = np.log(1 + comments_count)
                    history[parent_id][_id] = {
                        'relation': '{0:.4f}'.format(relation),
                        'comments_count': comments_count,
                        'normalized_comments_count': normalized_comments_count,
                        'reaction_count': reaction_count,
                        'normalized_reaction_count': normalized_reaction_count,
                        'datetime_pub': datetime_pub
                    }
                    properties['reaction_count'] += reaction_count
                    properties['comments_count'] += comments_count
                    properties['posts_count'] += 1                    
                    properties['score'] += (title_relation * alpha + relation * (1 - alpha)) * normalized_reaction_count

            if properties['score'] != 0:
                obj_score[obj['id']] = properties

    result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'], reverse=True)
    
    data = list()
    for obj, props in result:
        obj = find_fb_obj(obj)
        row = {
            'id': obj['id'],
            'name': obj['name'],
            'relation': '{0:.4f}'.format(props['relation']),
            'comments_count': props['comments_count'],
            'reaction_count': props['reaction_count'],
            'posts_count': props['posts_count'],
            'total_comments_count': props['total_comments_count'],
            'total_reaction_count': props['total_reaction_count'],
            'total_posts_count': props['total_posts_count'],
            'avg_comments_count': '{0:.2f}'.format(props['comments_count'] / props['posts_count']),
            'avg_reaction_count': '{0:.2f}'.format(props['reaction_count'] / props['posts_count'])
        }
        data.append(row)

    ret = json.dumps({'data': data, 'history': history}, default=json_util.default)
    print(time() - pt)
    return ret



@query_page.route('/ptt/', methods=['POST'])
def ptt_query():
    it = time()
    req = request.get_json()
    query_word = req['keyword']
    start = req['start']
    end = req['end']
    print(req)

    obj_score = defaultdict(lambda: defaultdict(int))
    history = defaultdict(dict)
    
    alpha = 0.5
    options = default_options(start, end)
    fields = {
        'uniID': 1,
        'title': 1,
        'datetime_pub': 1,
        'author': 1,
        'tfidf': 1,
        'url': 1,
        'comments_count': 1,
    }
    cnt = 0
    for post in ptt_posts.find(options, fields, no_cursor_timeout=True).sort([('datetime_pub', 1)]):
        if cnt % 10000 == 0:
            sys.stdout.write(f'\r{cnt}')
        cnt += 1
        
        uniID = post['uniID']
        author = post['author']
        title = post['title']
        url = post['url']
        comments_count = post['comments_count']
        datetime_pub = post['datetime_pub']

        obj_score[author]['id'] = author
        obj_score[author]['total_comments_count'] += comments_count
        obj_score[author]['total_posts_count'] += 1
        title_relation = 0 if query_word not in title else 0.5

        if 'tfidf' in post and query_word in dict(post['tfidf']):
            relation = alpha * title_relation + (1 - alpha) * dict(post['tfidf'])[query_word]
            normalized_comments_count = np.log(1 + comments_count)
            score = relation * normalized_comments_count
            history[author][uniID] = {
                'title': title,
                'relation': relation,
                'comments_count': comments_count,
                'normalized_comments_count': normalized_comments_count,
                'datetime_pub': datetime_pub,
                'url': url,
                'score': score
            }
            obj_score[author]['relation'] += relation
            obj_score[author]['comments_count'] += comments_count
            obj_score[author]['posts_count'] += 1                    
            obj_score[author]['score'] += score

    result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'] / (1 + np.log(1 + obj[1]['posts_count'])) * (obj[1]['posts_count'] / obj[1]['total_posts_count']), reverse=True)
    
    data = list()
    tmp = defaultdict(dict)
    for obj, props in result:
        if props['score'] > 0:
            tmp[obj] = history[obj]
            row = {
                'id': props['id'],
                'relation': '{0:.6f}'.format(props['relation'] / props['posts_count']),
                'comments_count': props['comments_count'],
                'posts_count': props['posts_count'],
                'total_comments_count': props['total_comments_count'],
                'total_posts_count': props['total_posts_count'],
                'avg_comments_count': '{0:.2f}'.format(props['comments_count'] / props['posts_count']),
                'score': props['score'] / (1 + np.log(1 + props['posts_count'])) * (props['posts_count'] / props['total_posts_count'] )
            }
            data.append(row)
    ret = json.dumps({'data': data, 'history': tmp}, default=json_util.default)
    print(f'\n{time() - it}')
    return ret


# @query_page.route('/fb/fulltext/', methods=['POST'])
# def fb_fulltext_query():
#     """
#     Get relative fb_objects by keyword.
#     ---
#     tags:
#         - query
#     parameters:
#         - name: payload
#           in: body
#           type: object
#           properties:
#               keyword:
#                   type: string
#               default: {"keyword":"化妝品", "start_date": "2018-01-01", "end_date": "2018-12-31"}
#     responses:
#         200:
#             ret: return value
#     """
#     it = time()
#     req = request.get_json()
#     query_word = req['keyword']
#     start = req['start']
#     end = req['end']
#     print(req)

#     obj_score = defaultdict(lambda: defaultdict(int))
#     history = defaultdict(dict)
#     alpha = 0.5
    
#     cnt = 0
#     for post in fb_posts_list:
#         if post['datetime_pub'] < datetime(2018, int(start), 1) or post['datetime_pub'] >= datetime(2018, int(end), 1): continue
#         if cnt % 10000 == 0:
#             sys.stdout.write(f"\r{cnt}")
#         cnt += 1
#         _id = post['id']
#         parent_id = post['parent_id']
#         parent_name = post['parent_name'] if post['parent_name'] != None else ''
#         comments_count = post['comments_count']
#         datetime_pub = post['datetime_pub']

#         obj_score[parent_id]['id'] = parent_id
#         obj_score[parent_id]['name'] = parent_name
#         obj_score[parent_id]['total_comments_count'] += comments_count
#         obj_score[parent_id]['total_posts_count'] += 1
#         title_relation = 0 if query_word not in parent_name else 1

#         if 'content' in post and str(post['content']) != 'nan':
#             words_freq = len(re.findall(f'{(query_word)}', post['content']))
#             if words_freq > 0:
#                 relation = words_freq
#                 normalized_comments_count = np.log(1 + post['comments_count'])
#                 score = relation * normalized_comments_count
#                 if score > 0:
#                     history[parent_id][_id] = {
#                         'relation': relation,
#                         'comments_count': comments_count,
#                         'normalized_comments_count': normalized_comments_count,
#                         'datetime_pub': datetime_pub,
#                         'score': score
#                     }
#                 obj_score[parent_id]['relation'] += relation
#                 obj_score[parent_id]['comments_count'] += comments_count
#                 obj_score[parent_id]['posts_count'] += 1                    
#                 obj_score[parent_id]['score'] += score

#     result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'] / (1 + np.log(1 + obj[1]['posts_count'])) * (obj[1]['posts_count'] / obj[1]['total_posts_count']), reverse=True)
#     data = list()
#     tmp = defaultdict(dict)
#     for obj, props in result:
#         if props["score"] > 0:
#             tmp[obj] = history[obj]
#             row = {
#                 'id': props['id'],
#                 'name': props['name'],
#                 'relation': '{0:.6f}'.format(props['relation'] / props['posts_count']),
#                 'score': props['score'] / (1 + np.log(1 + props['posts_count'])) * (props['posts_count'] / props['total_posts_count']),
#                 'comments_count': props['comments_count'],
#                 'posts_count': props['posts_count'],
#                 'total_comments_count': props['total_comments_count'],
#                 'total_posts_count': props['total_posts_count'],
#                 'avg_comments_count': '{0:.2f}'.format(props['comments_count'] / props['posts_count'])
#             }
#             data.append(row)
#     ret = json.dumps({'data': data, 'history': tmp}, default=json_util.default)
#     print(f'\n{time() - it}')
#     return ret
