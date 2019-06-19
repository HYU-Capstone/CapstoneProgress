import flask
from flask import request
from PIL import Image
import time
app = flask.Flask(__name__)


@app.route('/match', methods=['POST'])
def match():
    image = request.files['image']
    if image:
        with open('images/test{}.jpg'.format(time.time()), 'wb') as fw:
            fw.write(image.read())
    else:
        print('Image empty!')
    return 'OK'


app.run('0.0.0.0', port=5001)
