import pprint
import csv
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

remapping = [
    'Background',
    'Face',
    'Body',
    'Bicycle',
    'Car',
    'Motorbike',
    'Airplane',
    'Ship',
    'Bird',
    'Cat',
    'Dog',
    'Horse',
    'Cow'
]

collection = client['obj']['obj_256x256_20201102']
obj_count_in_single_img = [0] * 100
label_id_test = []
label_id_train = []

for data in tqdm(collection.find({'subset': 'val'})):
    label_id_test.extend(data['label_id'])

for data in tqdm(collection.find({'subset': 'train'})):
    label_id_train.extend(data['label_id'])

counter_label_id_test = dict(Counter(label_id_test))
counter_label_id_train = dict(Counter(label_id_train))

with open('statistics-info/obj-count.csv', 'w') as FILE:
    writer = csv.writer(FILE)
    writer.writerow(['Dataset'] + remapping[1:13])
    
    row = ['Testing Set']
    for i in range(1, 13):
        row.append(counter_label_id_test[i])

    writer.writerow(row)

    row = ['Training Set']
    for i in range(1, 13):
        row.append(counter_label_id_train[i])

    writer.writerow(row)