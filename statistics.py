import pprint
import csv
from pymongo import MongoClient
from collections import Counter
from tqdm import tqdm
from itertools import zip_longest

client = MongoClient(
    'localhost',
    username='mongoadmin',
    password='mongoadmin',
    authSource='admin',
    authMechanism='SCRAM-SHA-256'
)

collection = client['obj']['obj_256x256_20200930']
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

def area_statistics(label_id, subset='train'):
    area_interval = [0] * 12
    
    for data in collection.find({'subset': subset}):
        locs = zip_longest(data['xmin'], data['xmax'], data['ymin'], data['ymax'])
        
        for idx, (xmin, xmax, ymin, ymax) in enumerate(locs):
            area = (xmax - xmin) * (ymax - ymin)

            if(data['label_id'][idx] == label_id):
                if area < (0.05 * 0.05):
                    area_interval[0] += 1
                elif area < (0.1 * 0.1) and area >= (0.05 * 0.05):
                    area_interval[1] += 1
                elif area < (0.2 * 0.2) and area >= (0.1 * 0.1):
                    area_interval[2] += 1
                elif area < (0.3 * 0.3) and area >= (0.2 * 0.2):
                    area_interval[3] += 1
                elif area < (0.4 * 0.4) and area >= (0.3 * 0.3):
                    area_interval[4] += 1
                elif area < (0.5 * 0.5) and area >= (0.4 * 0.4):
                    area_interval[5] += 1
                elif area < (0.6 * 0.6) and area >= (0.5 * 0.5):
                    area_interval[6] += 1
                elif area < (0.7 * 0.7) and area >= (0.6 * 0.6):
                    area_interval[7] += 1
                elif area < (0.8 * 0.8) and area >= (0.7 * 0.7):
                    area_interval[8] += 1
                elif area < (0.9 * 0.9) and area >= (0.8 * 0.8):
                    area_interval[9] += 1
                elif area < (0.95 * 0.95) and area >= (0.9 * 0.9):
                    area_interval[10] += 1
                else:
                    area_interval[11] += 1

    return area_interval

with open('area_interval.csv', 'w') as FILE:
    writer = csv.writer(FILE)
    writer.writerow([
        '', 
        '> 0', 
        '> 0.05 * 0.05', 
        '> 0.1 * 0.1', 
        '> 0.2 * 0.2', 
        '> 0.3 * 0.3', 
        '> 0.4 * 0.4',
        '> 0.5 * 0.5',
        '> 0.6 * 0.6', 
        '> 0.7 * 0.7',
        '> 0.8 * 0.8',
        '> 0.9 * 0.9',
        '> 0.95 * 0.95'
    ])

    for i in tqdm(range(1, 13)):
        interval = area_statistics(i, subset='val')
        interval = [remapping[i]] + interval
        writer.writerow(interval)