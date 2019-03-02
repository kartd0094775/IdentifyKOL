import sys
import pymysql
import pymongo
import pandas as pd
from pymongo import MongoClient

from database import fb_database

client = MongoClient('localhost', 27017)
database = client['research']
fb_posts = database['2018_fb_posts']

def pad(month):
    return str(month).rjust(2, "0")

def dump_data(month):
    if month < 12:
        start = f"2018-{pad(month)}-01"
        end = f"2018-{pad(month+1)}-01"
    else:
        start = "2018-12-01"
        end = "2019-01-01"
    print(start, end)
    sql = f"SELECT * FROM fbposts WHERE `datetime_pub` >= '{start}' and `datetime_pub` < '{end}'"
    return pd.read_sql(sql, conn)

if __name__ == "__main__":
    # conn = pymysql.connect(**fb_database)
    # cursor = conn.cursor()
    # df = dump_data(2)
    # records= df.to_dict('records')
    # fb_posts.insert_many(records)
    # for i in range(6, 13):
    #     df = dump_data(i)
    #     records = df.to_dict('records')
    #     fb_posts.insert_many(records)