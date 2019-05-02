from collections import defaultdict
from time import sleep, time
import itertools
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

from synonyms import find_synonyms, is_in_vocab
from api import *
from preprocessing import *

ALPHA = 0.25
PARAM_K1 = 2.0
PARAM_B = 0.75
fb_avcl = (468271862) / 1547774
fb_avdl = (39839103 + 46215754) / (1491207 + 1547773)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def bm25(value, dl): 
    global PARAM_K1
    global PARAM_B
    global fb_avdl

    K = PARAM_K1 * (1 - PARAM_B + PARAM_B * (dl / fb_avdl))    
    return value * (PARAM_K1 + 1) / (value + K)

# def BM25F()

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

def fb_query(query_word, start=3, end=6, threshold=0.75):
    it = time()

    title_count = 0
    for obj in fb_objects_ref.values():
        if obj['name'] != None and query_word in obj['name']:
            title_count += 1
    title_count = float(np.log10( 9 + title_count))

    obj_score = defaultdict(lambda: defaultdict(int))
    history = defaultdict(dict)

    wordset =query_word.split('|')
    options = default_options(start, end)
    options['sentence'] = {'$ne': None}
    fields = default_fields()
    fields['words_count'] = 1
    fields['reaction_count'] = 1
    fields['attachments'] = 1
    fields['sentence'] = 1

    cnt = 0
    for post in fb_posts.find(options, fields, no_cursor_timeout=True).sort([('datetime_pub', 1)]):
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
    ret = {'data': data, 'history': tmp}
    print(f'\n{time() - it}')
    return ret

def ptt_query(query_word, start, end, threshold=0.75):
    it = time()

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
    ret = {'data': data, 'history': tmp}
    print(f'\n{time() - it}')
    return ret

def rank_keywords(fb_id, threshold):
    keywords = defaultdict(lambda: defaultdict(int))
    options = {'datetime_pub': {'$gte': datetime(2018, 3, 1), '$lt': datetime(2018, 6, 1)}}
    options['parent_id'] = fb_id
    options['sentence'] = {'$ne': None}
    total_posts_count = 0
    
    for post in fb_posts.find(options).sort([('datetime_pub', 1)]):
        sentence = post['sentence'].split()
        tokenized = list(itertools.chain.from_iterable(tokenize(sentence)))
        vocab = defaultdict(int)
        if len(tokenized) > 0:
            total_posts_count += 1
            for word in tokenized:
                vocab[word] = 1
            for word in vocab.keys():
                wordset = find_synonyms(word, threshold)
                word_class = " ".join(sorted(wordset.keys(), key=lambda x: x[0]))
                prog = re.compile( "|".join(list(map(lambda x: "(" + x + ")", list(wordset.keys())))) )
                freq = len(prog.findall(post['sentence']))
                
                relation = bm25(freq, len(tokenized))
                comments_count = post['comments_count']
                reaction_count = None if str(post['reaction_count']) == 'nan' else json.loads(post['reaction_count'])
                likes_count = 0 if reaction_count == None else int(np.sum(list(reaction_count.values())))

                keywords[word_class]['posts_count'] += 1 
                keywords[word_class]['relation'] += relation
                keywords[word_class]['likes_count'] += likes_count
                keywords[word_class]['comments_count'] += comments_count
                keywords[word_class]['score'] += relation * np.log(comments_count * 2 + likes_count + 1)
                
    idf = lambda x: np.log((total_posts_count - x + 0.5) / (x + 0.5) )
    print(total_posts_count)
    for word_class, props in keywords.items():
        props['score'] = props['score'] * idf(props['posts_count'])
        keywords[word] = props
    return sorted(keywords.items(), key=lambda x: x[1]['score'], reverse=True)


def rank_themes(req):

    # ### Usage Example ###
    # req = {
    #     'start': '2018-03-01',
    #     'end': '2018-06-01',
    #     'relation_boosts': {
    #         'title': 0.6,
    #         'article': 0.5,
    #         'comment': 0.1
    #     },
    #     'transformation': 'linear',
    #     'stats_type': 'both'
    # }
    # rank_themes(obj, themes, req)
    
    # Initialize parameter
    fb_id = req['fb_id']
    themes = req['themes']
    transformation = req['transformation']
    relation_boosts = req['relation_boosts']
    stats_type = req['stats_type']
    start = req['start']
    end = req['end']
    
    # Initialize varialbe
    theme_score = defaultdict(lambda: defaultdict(int))
    history = defaultdict(dict)
    total_posts_count = 0
    global_comments_count = list() # for normalization
    global_likes_count = list() # for normalization
    
    # MongoDB query options
    options = default_options(start, end)
    options['parent_id'] = fb_id  
    
    for post in fb_posts.find(options).sort([('datetime_pub', 1)]):
        
        # Post attributes
        _id = post['id']
        article_content = "" if 'sentence' not in post else post['sentence']
        comments_count = post['comments_count']
        comments_content = "" if 'comments_content' not in post else " ".join(post['comments_content'])
        reaction = {} if str(post['reaction_count']) == 'nan' else json.loads(post['reaction_count'])
        positive = 0 if len(reaction) == 0 else reaction['haha'] + reaction['love']
        negative = 0 if len(reaction) == 0 else reaction['angry'] + reaction['sad']
        likes_count = int(np.sum(list(reaction.values())))
        
        # Transformation
        if ( transformation == 'sqrt' ):
            trans_comments_count = np.sqrt(comments_count)
            trans_likes_count = np.sqrt(likes_count)
        elif ( transformation == 'log'):
            trans_comments_count = np.log1p(comments_count)
            trans_likes_count = np.log1p(likes_count)
        else:
            trans_comments_count = comments_count
            trans_likes_count = likes_count
            
        global_comments_count.append(trans_comments_count)
        global_likes_count.append(trans_likes_count)
        
        # Add total posts count
        total_posts_count += 1
        
        for query in themes:
            
            # Calculate relation
            wordset = query.split('|')  
            prog = re.compile( "|".join(list(map(lambda x: "(" + x + ")", wordset))) )
            fields = defaultdict(lambda: defaultdict(int))
            fields['article']['freq'] = len(prog.findall(article_content))
            fields['article']['dl'] = 0 if 'words_count' not in post else post['words_count']
            fields['article']['boost'] = float(relation_boosts['article'])
            fields['comment']['freq'] = len(prog.findall(" ".join(comments_content)))
            fields['comment']['dl'] = 0 if 'comments_words_count' not in post else post['comments_words_count']
            fields['comment']['boost'] = float(relation_boosts['comment'])
            relation = BM25F(fields)
            
            if ((comments_count > 0 or likes_count > 0) and relation > 0):
                history[query][_id] = {
                    'relation': relation,
                    'positive': positive,
                    'negative': negative,
                    'comments_count': comments_count,
                    'likes_count': likes_count,
                    'trans_comments_count': trans_comments_count,
                    'trans_likes_count': trans_likes_count,
                }
                
    global_comments_count = np.array(global_comments_count).astype(np.float32).reshape(-1, 1)
    global_likes_count = np.array(global_likes_count).astype(np.float32).reshape(-1, 1)

    comments_scaler = MinMaxScaler().fit(global_comments_count)
    likes_scaler = MinMaxScaler().fit(global_likes_count)    

    for query, doc in history.items():
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
            history[query][_id]['score'] = score
            history[query][_id]['normalized_comments_count'] = normalized_comments_count

            theme_score[query]['relation'] += relation
            theme_score[query]['likes_count'] += likes_count
            theme_score[query]['comments_count'] += comments_count
            theme_score[query]['positive'] += positive
            theme_score[query]['negative'] += negative
            theme_score[query]['posts_count'] += 1
            theme_score[query]['score'] += score
            
    # idf = lambda x: np.log((total_posts_count - x + 0.5) / (x + 0.5) )
    return sorted(theme_score.items(), key=lambda x: x[1]['score'] * x[1]['posts_count'], reverse=True)
        
        

def time_series(req):
    
    # Intialize parameter
    fb_id = req['fb_id']
    query_word = req['keywords']
    transformation = req['transformation']
    stats_type = req['stats_type']
    relation_boosts = req['relation_boosts']
    start = strptime(req['start'])
    end = strptime(req['end'])
    windows = req['windows']
    
    # Initialize variable
    date_score = defaultdict(lambda: defaultdict(int))
    history = defaultdict(lambda: defaultdict(int))
    total_posts_count = 0
    global_comments_count = list()
    global_likes_count = list()
    wordset = query_word.split('|')
    prog = re.compile( "|".join(list(map(lambda x: "(" + x + ")", wordset))) )
    
    # MongoDB query options
    options = {'datetime_pub': {'$gte': start - timedelta(days=windows), '$lt': end}}
    options['parent_id'] = fb_id

    for post in fb_posts.find(options).sort([('datetime_pub', 1)]):
    
        # Post attributes
        _id = post['id']
        datetime_pub = post['datetime_pub']
        article_content = "" if 'sentence' not in post else post['sentence']
        comments_count = post['comments_count']
        comments_content = "" if 'comments_content' not in post else " ".join(post['comments_content'])
        reaction = {} if str(post['reaction_count']) == 'nan' else json.loads(post['reaction_count'])
        positive = 0 if len(reaction) == 0 else reaction['haha'] + reaction['love']
        negative = 0 if len(reaction) == 0 else reaction['angry'] + reaction['sad']
        likes_count = int(np.sum(list(reaction.values())))
        
        
        # Transformation
        if ( transformation == 'sqrt' ):
            trans_comments_count = np.sqrt(comments_count)
            trans_likes_count = np.sqrt(likes_count)
        elif ( transformation == 'log'):
            trans_comments_count = np.log1p(comments_count)
            trans_likes_count = np.log1p(likes_count)
        else:
            trans_comments_count = comments_count
            trans_likes_count = likes_count
            
        global_comments_count.append(trans_comments_count)
        global_likes_count.append(trans_likes_count)
        
        # Add total posts count
        total_posts_count += 1
        
        
        fields = defaultdict(lambda: defaultdict(int))
        fields['article']['freq'] = len(prog.findall(article_content))
        fields['article']['dl'] = 0 if 'words_count' not in post else post['words_count']
        fields['article']['boost'] = float(relation_boosts['article'])
        fields['comment']['freq'] = len(prog.findall(" ".join(comments_content)))
        fields['comment']['dl'] = 0 if 'comments_words_count' not in post else post['comments_words_count']
        fields['comment']['boost'] = float(relation_boosts['comment'])
        relation = BM25F(fields)

        if ((comments_count > 0 or likes_count > 0) and relation > 0):
            history[_id] = {
                'datetime_pub': datetime_pub,
                'relation': relation,
                'positive': positive,
                'negative': negative,
                'comments_count': comments_count,
                'likes_count': likes_count,
                'trans_comments_count': trans_comments_count,
                'trans_likes_count': trans_likes_count,
        }
            
    global_comments_count = np.array(global_comments_count).astype(np.float32).reshape(-1, 1)
    global_likes_count = np.array(global_likes_count).astype(np.float32).reshape(-1, 1)

    comments_scaler = MinMaxScaler().fit(global_comments_count)
    likes_scaler = MinMaxScaler().fit(global_likes_count)             
            
    days = (end - start).days
    for i in range(days):
        cur = start + timedelta(i)
        for _id, props in history.items():
            if props['datetime_pub'] >= (cur - timedelta(days=windows)) and props['datetime_pub'] <= cur:
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
                
                date_score[cur.date()]['normalized_comments_count'] += normalized_comments_count
                date_score[cur.date()]['posts_count'] += 1
                date_score[cur.date()]['relation'] += relation
                date_score[cur.date()]['likes_count'] += likes_count
                date_score[cur.date()]['comments_count'] += comments_count
                date_score[cur.date()]['positive'] += positive
                date_score[cur.date()]['negative'] += negative
                date_score[cur.date()]['score'] += score
    return date_score, history