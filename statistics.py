import pprint
from pymongo import MongoClient
from collections import Counter
from tqdm import tqdm

client = MongoClient(
    'localhost',
    username='mongoadmin',
    password='mongoadmin',
    authSource='admin',
    authMechanism='SCRAM-SHA-256'
)

collection = client['obj']['obj_256x256_20200930']
obj_count_in_single_img = [0] * 100
label_id_total = []

for data in tqdm(collection.find({'subset': 'val'})):
    label_id_total.extend(data['label_id'])


print(Counter(label_id_total))