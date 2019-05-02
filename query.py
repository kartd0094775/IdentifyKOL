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
import gc
from pymongo import MongoClient
from bson import json_util
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from math import exp

from util.synonyms import find_synonyms, find_candidate, is_in_vocab
from util.api import *


query_page = Blueprint('query', 'query')

PARAM_K1 = 1.2
PARAM_B = 0.75


FB_AVG_LENGTH = {
    'title': (32739) / 1404,
    'article': (39839103 + 46215754) / (1491207 + 1547773),
    'comment': (24586783) / 1547774
}
fb_avtl = (32739) / 14041
# fb_avcl = (468271862) / 1547774
fb_avcl = (24586783) / 1547774
fb_avdl = (39839103 + 46215754) / (1491207 + 1547773)
fb_max_likes_count = 177250
fb_max_commnets_count = 82453
fb_total_likes_count = 233203525
fb_total_comments_count = 24586783

def preload(post_tag=False):
    fb_posts_list = list()
    if post_tag ==True:
        options = default_options(3, 6)
        # options['sentence'] = {'$ne': None}
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
    fields = default_fields()
    fields['sentence'] = 1
    fields['likes_count'] = 1
    fields['positive'] = 1
    fields['negative'] = 1
    print(options)
    cnt = 0
    start = strptime(start)
    end = strptime(end)
    # for post in fb_posts.find(options, fields, no_cursor_timeout=True):
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

        history[parent_id][_id] = {
            'relation': relation,
            'likes_count': likes_count,
            'comments_count': comments_count,
            'trans_comments_count': trans_comments_count,
            'trans_likes_count': trans_likes_count,
            'datetime_pub': datetime_pub,
            'positive': positive,
            'negative': negative,
        }

    return obj_score, history, global_comments_count, global_likes_count


@query_page.route('/similar/', methods=['POST'])
def most_similart_word():
    req = request.get_json()
    positives = req['keywords'].split(' ')
    candidates = find_candidate(positives)
    ret = json.dumps({'candidates': candidates}, default=json_util.default)
    return ret

@query_page.route('/fb/', methods=['POST'])
def fb_query():
    """
    Rank relative fb_objects by keyword.
    ---
    tags:
        - query
    parameters:
        - name: payload
          in: body
          type: object
          description: data payload
          properties:
              keywords:
                type: string
                example: "音樂"
              start:
                type: string
                example: "2018-04-01"
              end:
                type: string
                example: "2018-05-01"
              transformation:
                type: string
                enum: ["linear", "sqrt", "log"]
                example: "linear"
              relation_boosts:
                type: object
                properties:
                    title:
                        type: string
                        example: "0.3"
                    article:
                        type: string
                        example: "0.7"
                    comment:
                        type: string
                        example: "0.1"
              stats_type:
                type: string
                enum: ["both", "comment", "like"]
                example: "both"
              num:
                type: string
                example: "10"
    responses:
        200:
            ret: return value
    """
    it = time()
    req = request.get_json()
    print(req)
    query_word = req['keywords']
    transformation = req['transformation']
    relation_boosts = req['relation_boosts']
    stats_type = req['stats_type']
    start = req['start']
    end = req['end']
    num = int(req['num'])

    try: 
        wordset = query_word.split(' ')
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

            if (stats_type == 'like'): score = relation * normalized_likes_count
            elif (stats_type == 'comment'): score = relation * normalized_comments_count
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
            if len(data) == num: break

    ret = json.dumps({'data': data, 'history': tmp}, default=json_util.default)
    print(f'\n{time() - it}')
    gc.collect()
    return ret

@query_page.route('fb_theme/', methods=['POST'])
def fb_themes():
    # def rank_themes(fb_id, themes, threshold=0.75):
    theme_score = defaultdict(lambda: defaultdict(int))
    
    options = {'datetime_pub': {'$gte': datetime(2018, 3, 1), '$lt': datetime(2018, 6, 1)}}
    options['parent_id'] = fb_id
    options['sentence'] = {'$ne': None}
    total_posts_count = 0
    
    for post in fb_posts.find(options).sort([('datetime_pub', 1)]):
        sentence = post['sentence'].split()
        tokenized = list(itertools.chain.from_iterable(tokenize(sentence)))
        if len(tokenized) > 0:
            total_posts_count += 1
            for query in themes:
                wordset = query.split('|')   
                if len(wordset) > 1:
                    word_class = " ".join(sorted(wordset, key=lambda x: x[0]))
                else:
                    word_class = wordset[0]
                prog = re.compile( "|".join(list(map(lambda x: "(" + x + ")", wordset))) )
                freq = len(prog.findall(post['sentence']))
                if freq > 0:
                    relation = bm25(freq, len(tokenized))
                    comments_count = post['comments_count']
                    reaction_count = None if str(post['reaction_count']) == 'nan' else json.loads(post['reaction_count'])
                    likes_count = 0 if reaction_count == None else int(np.sum(list(reaction_count.values())))

                    theme_score[word_class]['posts_count'] += 1 
                    theme_score[word_class]['relation'] += relation
                    theme_score[word_class]['likes_count'] += likes_count
                    theme_score[word_class]['comments_count'] += comments_count
                    theme_score[word_class]['score'] += relation * (comments_count * 2 + likes_count)
    
    for word_class, props in theme_score.items():
        theme_score[word_class]['comments_count'] = np.log(props['comments_count'])
    idf = lambda x: np.log((total_posts_count - x + 0.5) / (x + 0.5) )
    return sorted(theme_score.items(), key=lambda x: x[1]['score'], reverse=True)
        

@query_page.route('/ptt/', methods=['POST'])
def ptt_query():
    it = time()
    req = request.get_json()
    query_word = req['keyword']
    start = int(req['start'])
    end = int(req['end'])
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
            score = relation * comments_count
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

    result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'] * (obj[1]['posts_count'] / obj[1]['total_posts_count']), reverse=True)
    
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