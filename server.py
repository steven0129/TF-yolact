import pprint
import gridfs
import io

from PIL import Image
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

client = MongoClient(
    'localhost',
    username='mongoadmin',
    password='mongoadmin',
    authSource='admin',
    authMechanism='SCRAM-SHA-256'
)

collection = client['obj']['obj_256x256_20200921']
fs = gridfs.GridFS(client['obj'])

records = list(collection.find())

@app.route('/img', methods=['GET'])
def img():
    recordID = request.args.get('id')
    imgID = records[int(recordID)]['image']
    img = io.BytesIO(fs.get(ObjectId(imgID)).read())
    return send_file(img, mimetype='image/jpg')

@app.route('/data', methods=['GET'])
def data():
    recordID = request.args.get('id')
    record = records[int(recordID)]

    record['_id'] = str(record['_id'])
    record['image'] = str(record['image'])
    record['masks'] = list(map(str, record['masks']))

    return record

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)