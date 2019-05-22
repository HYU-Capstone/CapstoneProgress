from flask import Flask, send_from_directory, redirect, render_template, request, jsonify
from werkzeug.utils import secure_filename
from functions import *
from facenet import facenet
import tensorflow as tf
import math
import os
import numpy as np
import pickle
from sklearn.svm import SVC
from tensorflow.python.platform import gfile
import uuid
from threading import Thread, Lock
UPLOAD_FOLDER = './uploads'

data_dir = 'dataset'
model_path = 'models/20180402-114759.pb'
batch_size = 90
image_size = 160
classifier_filename = 'models/20190523.clf'

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

img_stack = []
mutex = Lock()

class RequestFailError(Exception):
  pass

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == 'jpg'

classifier_filename_exp = os.path.expanduser(classifier_filename)
with open(classifier_filename_exp, 'rb') as infile:
  loaded = pickle.load(infile)
model, class_names = loaded
print('Loaded classifier model from file "%s"' % classifier_filename_exp)

graph = tf.Graph()

model_exp = os.path.expanduser(model_path)
if os.path.isfile(model_exp):
  print('Model filename: %s' % model_exp)
  with gfile.FastGFile(model_exp,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def, name='')
  sess = tf.Session(graph=graph)
else:
  print('Model directory: %s' % model_exp)
  meta_file, ckpt_file = facenet.get_model_filenames(model_exp)
  
  print('Metagraph file: %s' % meta_file)
  print('Checkpoint file: %s' % ckpt_file)

  sess = tf.Session(graph=graph)
  saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
  saver.restore(sess, os.path.join(model_exp, ckpt_file))


@app.route('/static/js/<path:path>')
def serve_static_js(path):
  return send_from_directory('static/js', path)

@app.route('/static/css/<path:path>')
def serve_static_css(path):
  return send_from_directory('static/css', path)

@app.route('/static/img/<path:path>')
def serve_static_img(path):
  return send_from_directory('static/img', path)

@app.route('/static/fonts/<path:path>')
def serve_static_fonts(path):
  return send_from_directory('static/fonts', path)

@app.route('/')
def redirect_to_users():
  return redirect('/users')

@app.route('/users')
def users():
  return render_template('users.j2')

@app.route('/users/<user_id>')
def user(user_id):
  return render_template('user.j2', user_id=user_id)

@app.route('/attendances')
def attendances():
  return render_template('attendances.j2')

@app.route('/api/users', methods=['GET', 'POST'])
def api_users():
  try:
    if request.method == 'GET':
      return jsonify({
        'success': True,
        'data': [o.__dict__ for o in get_users()]
      })
    elif request.method == 'POST':
      body = request.json()
      return jsonify({
        'success': True,
        'user_id': create_user(body['name'], body['email'])
      })
  except Exception as e:
    print(e)
    return jsonify({
      'success': False,
      'reason': str(e)
    })

@app.route('/api/users/<user_id>', methods=['GET', 'PUT'])
def api_user(user_id):
  try:
    if request.method == 'GET':
      return jsonify({
        'success': True,
        'data': get_user(user_id).__dict__
      })
    else:
      update_user(user_id, request.json())
      return jsonify({
        'success': True
      })
  except Exception as e:
    print(e)
    return jsonify({
      'success': False,
      'reason': str(e)
    })


@app.route('/api/attendances')
def api_attendances():
  try:
    return jsonify({
      'success': True,
      'data': [o.__dict__ for o in get_attendances()]
    })
  except Exception as e:
    print(e)
    return jsonify({
      'success': False,
      'reason': str(e)
    })

@app.route('/api/attendance/<id>')
def api_attendance(id):
  try:
    return jsonify({
      'success': True,
      'data': get_attendance(id).__dict__
    })
  except Exception as e:
    print(e)
    return jsonify({
      'success': False,
      'reason': str(e)
    })

@app.route('/api/classify', methods=['POST'])
def api_classify():
  try:
    if 'file' not in request.files:
      raise RequestFailError('File not provided')
    file = request.files['file']
    if file.filename == '':
      raise RequestFailError('No file selected')
    if file and allowed_file(file.filename):
      filename = secure_filename(str(uuid.uuid4()) + '.jpg')
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(filepath)
      img_stack.append(filepath)
      if len(img_stack) == 4:
        p = Thread(target=classify_process, args=(img_stack[:], sess, graph, class_names))
        p.start()
        img_stack.clear()
      return jsonify({
        'success': True
      })
    else:
      raise RequestFailError('Invalid file uploaded')
  except Exception as e:
    print(e)
    return jsonify({
      'success': False,
      'reason': str(e)
    })

def classify_process(imagepaths, sess, graph, class_names):
  with graph.as_default():
    # Get input and output tensors
    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    images = facenet.load_data(imagepaths, False, False, image_size)
    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
    emb_array = sess.run(embeddings, feed_dict=feed_dict)

    print('TF Session done')

    predictions = model.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    
    print('Prediction done')

    for index, probability in zip(best_class_indices, best_class_probabilities):
      print(class_names[index], probability)


if __name__ == '__main__':
  app.run(host='0.0.0.0')