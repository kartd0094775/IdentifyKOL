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
from api import *

# def fb_query(query_word, start, end):
#     pt = time()
#     obj_score = dict()
#     history = defaultdict(dict)
#     alpha = 0.5
    
#     for obj in fb_objects.find({'tfidf': {"$ne": None}}):
#         tfidf = dict(obj['tfidf'])
#         name = obj['name']
#         title_relation = 0 if query_word not in name else 1

#         if query_word in tfidf:
#             # relation, comments_count, posts_count, score
#             properties = defaultdict(int)
#             properties['relation'] = tfidf[query_word]

#             options = default_options(start, end)
#             options['parent_id'] = str(obj['id'])
#             field = default_field()
            
#             for post in fb_posts.find(options, field, no_cursor_timeout=True).sort([('datetime_pub', 1)]):
#                 _id = post['id']
#                 parent_id = post['parent_id']
#                 comments_count = post['comments_count']
#                 datetime_pub = post['datetime_pub']
#                 properties['total_comments_count'] += comments_count
#                 properties['total_posts_count'] += 1
#                 if 'global_tfidf' in post and query_word in dict(post['global_tfidf']):
#                     relation = dict(post['global_tfidf'])[query_word]
#                     normalized_comments_count = np.log(1 + comments_count)
#                     history[parent_id][_id] = {relation, comments_count, normalized_comments_count, datetime_pub}
#                     properties['comments_count'] += comments_count
#                     properties['posts_count'] += 1                    
#                     properties['score'] += (title_relation * alpha + relation * (1 - alpha)) * normalized_comments_count
                    
#             if properties['score'] != 0:
#                 obj_score[obj['id']] = properties

#     result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'], reverse=True)
    
#     data = list()
#     for obj, props in result:
#         obj = find_fb_obj(obj)
#         row = {
#             'id': obj['id'],
#             'name': obj['name'],
#             'relation': props['relation'],
#             'comments_count': props['comments_count'],
#             'posts_count': props['posts_count'],
#             'total_comments_count': props['total_comments_count'],
#             'total_posts_count': props['total_posts_count'],
#             'avg': '{0:.2f}'.format(props['comments_count'] / props['posts_count'])
#         }
#         data.append(row)
#     print(time() - pt)
#     return data, history

def fb_query(query_word, start, end):
    pt = time()
    obj_score = defaultdict(lambda: defaultdict(int))
    history = defaultdict(dict)
    alpha = 0.5

    options = default_options(start, end)
    field = default_field()
    
    for post in fb_posts.find(options, field, no_cursor_timeout=True).sort([('datetime_pub', 1)]):
        _id = post['id']
        parent_id = post['parent_id']
        parent_name = find_fb_obj[str(parent_id)]['name']
        comments_count = post['comments_count']
        datetime_pub = post['datetime_pub']

        obj_score['total_comments_count'] += comments_count
        properties['total_posts_count'] += 1
        title_relation = 0 if query_word not in parent_name else 1

        if 'global_tfidf' in post and query_word in dict(post['global_tfidf']):
            relation = dict(post['global_tfidf'])[query_word]
            normalized_comments_count = np.log(1 + comments_count)
            history[parent_id][_id] = {relation, comments_count, normalized_comments_count, datetime_pub}
            properties['comments_count'] += comments_count
            properties['posts_count'] += 1                    
            properties['score'] += (title_relation * alpha + relation * (1 - alpha)) * normalized_comments_count
            
    if properties['score'] != 0:
        obj_score[obj['id']] = properties

    result = sorted(obj_score.items(), key=lambda obj: obj[1]['score'], reverse=True)
    
    data = list()
    for obj, props in result:
        obj = find_fb_obj(obj)
        row = {
            'id': obj['id'],
            'name': obj['name'],
            'relation': props['relation'],
            'comments_count': props['comments_count'],
            'posts_count': props['posts_count'],
            'total_comments_count': props['total_comments_count'],
            'total_posts_count': props['total_posts_count'],
            'avg': '{0:.2f}'.format(props['comments_count'] / props['posts_count'])
        }
        data.append(row)
    print(time() - pt)
    return data, history

def fb_reaction_query(query_word, start, end):
    pt = time()
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
            field = default_field()
            field['reaction_count'] = 1

            for post in fb_posts.find(options, field, no_cursor_timeout=True).sort([('datetime_pub', 1)]):
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

    print(time() - pt)
    return data, history