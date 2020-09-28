import pprint
import gridfs
import io

from PIL import Image
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask import Flask, request, send_file

app = Flask(__name__)

client = MongoClient(
    'localhost',
    username='mongoadmin',
    password='mongoadmin',
    authSource='admin',
    authMechanism='SCRAM-SHA-256'
)

collection = client['obj']['obj_256x256_20200921']
fs = gridfs.GridFS(client['obj'])

@app.route('/img', methods=['GET'])
def img():
    imgID = request.args.get('id')
    img = io.BytesIO(fs.get(ObjectId(imgID)).read())
    return send_file(img, mimetype='image/jpg')

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)