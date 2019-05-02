from collections import defaultdict
from IPython import embed
from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep
from os import walk
import pandas as pd
import numpy as np
import pymongo
import json
import pymysql
from bson import json_util
from pymongo import MongoClient
from time import mktime, time, sleep
from datetime import datetime

from flask import Blueprint
from flask import request

from util.api import *

scape_page = Blueprint('scape', 'scape')


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
    start = req['start']
    end = req['end']
    
    history = dict()

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
            posts.append({
                'id': post['id'],
                'relation': 'NULL',
                'likes_count': post['likes_count'],
                'comments_count': post['comments_count'],
                'score': 'NULL',
                'datetime_pub': post['datetime_pub']
            })
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

    data = list()
    url = "https://www.facebook.com/search/pages/?q={}&epa=SERP_TAB".format(query_word)
    print(url)

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
        sleep(3)
        elements = driver.find_elements_by_css_selector('div > div._4bl9 > div > div._5aj7 > div._4bl9 > div > div > div > div > a')
        for i in range(num):
            element = elements[i]
            data.append({"name": element.text, "link": element.get_attribute('href')})
    except Exception as e:
        print(e)
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

    data = list()
    url = "https://www.google.com.tw/search?q=site:www.facebook.com+{}&source=lnt&tbs=cdr:1,cd_min:{},cd_max:{}&tbm=".format(query_word, start.strftime("%m/%d/%Y"), end.strftime("%m/%d/%Y"))
    print(url)

    try:
        option = webdriver.ChromeOptions()
        driver = webdriver.Chrome(options=option)
        driver.get(url)
        sleep(3)
        elements = driver.find_elements_by_css_selector(f'#rso > div > div div > div > div > div.r > a')
        for element in elements:
            name = element.text.split('\n')[0]
            link = element.get_attribute('href')
            data.append({"name": name, "link": link})
    except Exception as e:
        print(e)
    finally:
        driver.close()
    ret = json.dumps({'data': data})
    print(time() - it)
    return ret