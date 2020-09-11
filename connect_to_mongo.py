import json
import pymongo

client = pymongo.MongoClient('localhost')
db = client['mydb'] #db's name
collection = db['Hot'] #colletions'name


str = open("E:\google drive\iii_course\scrapy\hot_scrapy\hot_scrapy.json", encoding="utf-8-sig").read()
data = []
data.extend(json.loads(str))
collection.insert_many(data)
client.close()