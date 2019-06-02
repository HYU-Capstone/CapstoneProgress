from flask import Flask, send_from_directory, redirect, render_template, request, jsonify
from werkzeug.utils import secure_filename
from functions import *
import math
import json
import os
import sys
import numpy as np
import time
import pickle
import uuid
from multiprocessing import Process, Lock
# from threading import Thread, Lock
import shutil
from test_model import *
UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

img_stack = []

if not os.path.isdir(data_dir):
  os.mkdir(data_dir)

class RequestFailError(Exception):
  pass

def allowed_file(filename):
    return '.' in filename and \
          filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg']

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

@app.route('/users/new')
def new_user():
  return render_template('newuser.j2')

@app.route('/attendances')
def attendances():
  return render_template('attendances.j2')

@app.route('/dashboard')
def show_dashboard():
  return render_template('dashboard_live.j2')

@app.route('/api/users', methods=['GET', 'POST'])
def api_users():
  try:
    if request.method == 'GET':
      return jsonify({
        'success': True,
        'data': [o.__dict__ for o in get_users()]
      })
    elif request.method == 'POST':
      body = request.json
      return jsonify({
        'success': True,
        'user_id': create_user(body['name'], body['email'])
      })
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
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
      update_user(user_id, request.json)
      return jsonify({
        'success': True
      })
  except Exception as e:
    print(e)
    return jsonify({
      'success': False,
      'reason': str(e)
    })

@app.route('/api/users/<user_id>/train', methods=['POST'])
def api_train(user_id):
  try:
    user = get_user(user_id)
    file_keys = ['file' + str(x) for x in range(1, 6)]
    if 'file' not in request.files:
      raise RequestFailError('file not provided')
    file = request.files['file']
    if file:
      secured_filename = secure_filename(str(time.time()) + '.jpg')
      if not os.path.isdir('raw_trainset/' + str(user.user_id)):
        os.mkdir('raw_trainset/' + str(user.user_id))
      filepath = 'raw_trainset/{}/{}'.format(user.user_id, secured_filename)
      file.save(filepath)
    
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
    if file:
      filename = secure_filename(str(time.time()) + '.jpg')
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(filepath)
      img_stack.append(filepath)
      return jsonify({
        'success': True
      })
    else:
      raise RequestFailError('Invalid file uploaded - ' + file.filename)
  except Exception as e:
    print(e)
    return jsonify({
      'success': False,
      'reason': str(e)
    })


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
  # socketio.run(app)