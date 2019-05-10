from collections import defaultdict
from IPython import embed
from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep
from os import walk
import re
import dill as pkl
import pandas as pd
import numpy as np
import pymongo
import json
import pymysql
import hashlib
import traceback
import sys
from bson import json_util
from pymongo import MongoClient
from time import mktime, time, sleep
from datetime import datetime

from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from flask import Blueprint, request, abort

from util.api import *

scape_page = Blueprint('scape', 'scape')


PARAM_K1 = 1.2
PARAM_B = 0.75


FB_AVG_LENGTH = {
    'title': (32739) / 1404,
    'article': (39839103 + 46215754) / (1491207 + 1547773),
    'comment': (24586783) / 1547774
}

PTT_AVG_LENGTH = {
    'title': 10838625 / 2632314,
    'article': 210163557 / 2632314,
    'comment': 180204594 / 2632314
}


def BM25F(fields):
    global PARAM_K1
    global PARAM_B

    weight = 0
    for key, props in fields.items():
        weight += ( props['freq'] * props['boost'] ) / (1 - PARAM_B + PARAM_B * (props['dl'] / FB_AVG_LENGTH[key]))
    return weight / (PARAM_K1 + weight)


def gen_signature(req):
        m = hashlib.md5()
        signiture = ''
        for key, value in req.items():
            if key == 'relation_boosts':
                signiture += " ".join(list(value.values()))
            elif key == 'keywords':
                keywords = value.split(' ')
                keywords = sorted(keywords)
                signiture += " ".join(keywords)
            elif key != 'result' and key !='rate':
                signiture += str(value)
        m.update(signiture.encode('utf-8'))
        return m.hexdigest()

def calculate_one_obj(fb_id, keywords, start, end):
    options = default_options(start, end)
    options['parent_id'] = fb_id
    posts = list()
    for post in fb_posts.find(options):
        posts.append({
            'id': post['id'],
            'relation': 'NULL',
            'likes_count': post['likes_count'],
            'comments_count': post['comments_count'],
            'score': 'NULL',
            'datetime_pub': post['datetime_pub']
        })

@scape_page.route('/object/', methods=['POST'])
def object_scape():
    """
    Scape facebook result.
    ---
    tags:
        - scape
    parameters:
        - name: payload
          in: body
          type: object
          description: data payload
          properties:
              url:
                type: string
                example: "https://www.facebook.com/DoctorKoWJ/"
              start:
                type: string
                example: "2018-04-01"
              end:
                type: string
                example: "2018-05-01"
    responses:
        200:
            ret: return value
    """
    it = time()
    req = request.get_json()
    url = req['url']
    query_word = req['keywords']
    transformation = req['transformation']
    relation_boosts = req['relation_boosts']
    stats_type = req['stats_type']
    start = req['start']
    end = req['end']
    
    try: 
        wordset = query_word.split(' ')
        prog = re.compile( "|".join(list(map(lambda x: "(" + x + ")", wordset))) )
    except:
        print('[object_scape] query_word error: ' + query_word)
        prog = re.compile("(Empty)")


    history = dict()
    global_comments_count = list()
    global_likes_count = list()

    try:
        option = webdriver.ChromeOptions()
        driver = webdriver.Chrome(options=option)
        driver.get('https://www.facebook.com.tw/')
        element = driver.find_element_by_name('email')
        element.send_keys('tiy43511802000@yahoo.com')
        element = driver.find_element_by_name('pass')
        element.send_keys('fine019game087')
        driver.find_element_by_css_selector('#loginbutton > input').click()
        sleep(3)
        driver.get(url)
        element = driver.find_element_by_css_selector('div > div > div > a[aria-label="大頭貼照"]')
        fb_id = element.get_attribute('href').split('/')[3]
        element = driver.find_element_by_css_selector('#seo_h1_tag > a > span')
        name = element.text
        options = default_options(start, end)
        options['parent_id'] = fb_id
        posts = list()
        for post in fb_posts.find(options):
            _id = post['id']
            parent_id = post['parent_id']
            parent_name = post['parent_name']
            article_content = post['sentence']
            comments_count = post['comments_count']
            comments_content = post['comments_content']
            likes_count = post['likes_count']
            datetime_pub = post['datetime_pub']

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
            if float(relation_boosts['title']) != 0.0:
                fields['title']['freq'] = len(prog.findall(parent_name))
                fields['title']['dl'] = post['parent_words_count']
                fields['title']['boost'] = float(relation_boosts['title'])
            if float(relation_boosts['article']) != 0.0:
                fields['article']['freq'] = len(prog.findall(article_content))
                fields['article']['dl'] = post['words_count']
                fields['article']['boost'] = float(relation_boosts['article'])
            if float(relation_boosts['comment']) != 0.0:
                fields['comment']['freq'] = len(prog.findall(comments_content))
                fields['comment']['dl'] = post['comments_words_count']
                fields['comment']['boost'] = float(relation_boosts['comment'])
            relation = BM25F(fields)

            posts.append({
                'id': _id,
                'relation': relation,
                'likes_count': likes_count,
                'comments_count': comments_count,
                'trans_comments_count': trans_comments_count,
                'trans_likes_count': trans_likes_count,
                'score': 'NULL',
                'datetime_pub': datetime_pub
            })
        if len(posts) > 0:
            global_comments_count = np.array(global_comments_count).astype(np.float32).reshape(-1, 1)
            global_likes_count = np.array(global_likes_count).astype(np.float32).reshape(-1, 1)
            
            comments_scaler = MinMaxScaler().fit(global_comments_count)
            likes_scaler = MinMaxScaler().fit(global_likes_count)    

            for i, props in enumerate(posts):
                relation = props['relation']
                normalized_comments_count = comments_scaler.transform(np.array(props['trans_comments_count']).astype(np.float32).reshape(-1, 1))[0][0]
                normalized_likes_count = likes_scaler.transform(np.array(props['trans_likes_count']).astype(np.float32).reshape(-1, 1))[0][0]

                if (stats_type == 'like'): score = relation * normalized_likes_count
                elif (stats_type == 'comment'): score = relation * normalized_comments_count
                else: score = relation * (normalized_comments_count + normalized_likes_count)
                posts[i]['score'] = float(score)


        history['id'] = fb_id
        history['name'] = name
        history['posts'] = posts
    except Exception as e:
        print(e)
    finally:
        driver.close()
    ret = json.dumps({'history': history}, default=json_util.default)
    print(time() - it)
    return ret

@scape_page.route('/load_score/', methods=['POST'])
def load_score():
    it = time()
    req = request.get_json()
    data = req['result']
    del req['result']
    sig = gen_signature(req)
    print(f'[load_score] {req}')
    print(f'[load_score] {sig}')

    main_score = dict()
    main_history = dict()
    try:
        with open(f'./result/{sig}.pkl', 'rb') as f:
            result = pkl.load(f)
            for props in result['data']:
                main_score[props['id']] = props['score']
            main_history = result['history']
            for i, row in enumerate(data):
                fb_id = data[i]['id']
                if fb_id in main_score:
                    data[i]['score'] = main_score[fb_id]
                else:
                    data[i]['score'] = '粉絲團或貼文未被列入爬蟲資料庫中'
    except Exception as e:
        print(e)
        print('[load_score] main_sys not query yet!')
        return json.dumps({'data': data, 'message': '無法載入系統分數'})

    ret = json.dumps({'data': data}, default=json_util.default)
    print(time() - it)
    return ret

@scape_page.route('/fb/', methods=['POST'])
def fb_scape():
    """
    Scape facebook result.
    ---
    tags:
        - scape
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
              num:
                type: string
                example: "10"
    responses:
        200:
            ret: return value
    """
    it = time()
    req = request.get_json()
    query_word = req['keywords']
    num = int(req['num'])
    sig = gen_signature(req)
    print(f'[fb_scape] {req}')
    print(f'[fb_scape] {sig}')

    try:
        with open(f'./result/{sig}.pkl', 'rb') as f:
            ret = pkl.load(f)
            ret = json.dumps(ret)
            print(f'\n{time() - it}')
            return ret
    except:
        print('[fb_scape] facebook first query!')

    data = list()
    url = "https://www.facebook.com/search/pages/?q={}&epa=SERP_TAB".format(query_word)
    print(f'[fb_scape] {url}')


    try:
        option = webdriver.ChromeOptions()
        prefs = {"profile.default_content_setting_values.notifications" : 2}
        option.add_experimental_option("prefs",prefs)
        driver = webdriver.Chrome(options=option)
        driver.get('https://www.facebook.com.tw/')
        element = driver.find_element_by_name('email')
        element.send_keys('tiy43511802000@yahoo.com')
        element = driver.find_element_by_name('pass')
        element.send_keys('fine019game087')
        driver.find_element_by_css_selector('#loginbutton > input').click()
        sleep(1)
        driver.get(url)
        sleep(1)
        for i in range(15):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(1)

        elements = driver.find_elements_by_css_selector('div > div._4bl9 > div > div._5aj7 > div._4bl9 > div > div > div > div > a')
        while len(elements) < num:
            last_num = len(elements)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(1)
            elements = driver.find_elements_by_css_selector('div > div._4bl9 > div > div._5aj7 > div._4bl9 > div > div > div > div > a')
            if last_num == len(elements):
                print(f'[fb_scape] only have {last_num} elements')
                break

        last_num = num if len(elements) > num else len(elements) 
        for i in range(last_num):
            element = elements[i]
            name = element.text
            link = element.get_attribute('href')
            data.append({'name': name, 'link': link})

        for i, row in enumerate(data):
            name = row['name']
            link = row['link']
            obj = fb_objects.find_one({'name': name})
            if obj != None:
                driver.get(link)
                try:
                    obj_element = driver.find_element_by_css_selector('div > div > div > a[aria-label="大頭貼照"]')
                    fb_id = obj_element.get_attribute('href').split('/')[3]
                except:
                    fb_id = obj['id']
                score = "尚未載入主系統分數"
            else:
                fb_id = "null"
                score = '粉絲團未收入爬蟲資料庫中'
            data[i]['id'] = fb_id
            data[i]['score'] = score

        with open(f'./result/{sig}.pkl', 'wb') as f:
            pkl.dump({'data': data}, f)

    except Exception as e:
        print(e)
        print(sys.exc_info()[0:2])
        print(traceback.extract_tb(sys.exc_info()[2]))
    finally:
        driver.close()
    ret = json.dumps({'data': data})

    print(time() - it)
    return ret

@scape_page.route('/google/', methods=['POST'])
def google_scape():
    """
    Scape google result.
    ---
    tags:
        - scape
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
              num:
                type: string
                example: "10"
    responses:
        200:
            ret: return value
    """
    it = time()
    req = request.get_json()
    query_word = req['keywords']
    start = strptime(req['start'])
    end = strptime(req['end'])
    num = int(req['num'])
    sig = gen_signature(req)

    print(f'[google_scape] {req}')
    print(f'[google_scape] {sig}')

    try:
        with open(f'./result/{sig}.pkl', 'rb') as f:
            ret = pkl.load(f)
            ret = json.dumps(ret)
            print(f'\n{time() - it}')
            return ret
    except:
        print('[google_scape] google first query!')

    data = list()
    url = "https://www.google.com.tw/search?q=site:www.facebook.com+{}&source=lnt&tbs=cdr:1,cd_min:{},cd_max:{}&tbm=".format(query_word, start.strftime("%m/%d/%Y"), end.strftime("%m/%d/%Y"))
    print(f'[google_scape] {url}')

    try:
        option = webdriver.ChromeOptions()
        driver = webdriver.Chrome(options=option)
        driver.get(url)

        while len(data) < num:
            sleep(1)
            elements = driver.find_elements_by_css_selector(f'#rso > div > div div > div > div > div.r > a')
            for i, element in enumerate(elements):    
                name = element.text.split('\n')[0]
                link = element.get_attribute('href')
                if name != '轉為繁體網頁':
                    data.append({"name": name, "link": link})
                if i == len(elements) - 1:
                    driver.find_element_by_css_selector('#pnnext').click()
                if len(data) == num: break

        with open(f'./result/{sig}.pkl', 'wb') as f:
            pkl.dump({'data': data}, f)
            
        # for i, row in enumerate(data):
        #     driver.get(row['link'])
        #     sleep(1)
        #     try:
        #         element = driver.find_element_by_css_selector('#seo_h1_tag > a')
        #         data[i]['name'] = element.text
        #         data[i]['link'] = element.get_attribute('href')
        #         element = driver.find_element_by_css_selector('div > div > div > a[aria-label="大頭貼照"]')
        #         fb_id = element.get_attribute('href').split('/')[3]
        #         data[i]['id'] = fb_id
        #     except:
        #         data[i]['id'] = 'null'
        #         print("google_scape api:inner parse")
        #         print(sys.exc_info()[0:2])
        #         print(traceback.extract_tb(sys.exc_info()[2]))

    except Exception as e:
        print('google_scape api:')
        print(sys.exc_info()[0:2])
        print(traceback.extract_tb(sys.exc_info()[2]))
    finally:
        driver.close()
    ret = json.dumps({'data': data})


    print(time() - it)
    return ret