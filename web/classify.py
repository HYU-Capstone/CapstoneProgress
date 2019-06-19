import os
import pickle
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.python.platform import gfile
import socketio
from flask import Flask
import json 
from functions import * 
from threading import Thread, Lock
import time
import shutil
import math 
from aiohttp import web
import asyncio
from test_model import create_label_to_class_map, create_data_dir_result_label_map
from facenet import facenet

UPLOAD_FOLDER = './uploads'

sio = socketio.Server(async_mode='threading')
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

data_dir = 'dataset'
model_path = 'models/20180402-114759.pb'
batch_size = 90
image_size = 160
classifier_filename = 'models/' + sorted(filter(lambda x: 'clf' in x, os.listdir('models')))[-1]

mutex = Lock()

if os.path.isdir('tmp'):
  shutil.rmtree('tmp')
os.mkdir('tmp')

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


def background_task():
  while True:
    files = list(map(lambda x: os.path.join(UPLOAD_FOLDER, x), sorted(os.listdir(UPLOAD_FOLDER))))
    if len(files) >= 5:
      files = files[:5]
      sess_name = str(time.time())
      sess_dir = 'tmp/' + sess_name
      print('Opening session', sess_dir)
      os.mkdir(sess_dir)
      for filepath, index in zip(files, range(1, 6)):
        filename = str(index) + '.jpg'
        shutil.move(filepath, os.path.join(sess_dir, filename))
      t = Thread(target=classify_process, args=(sess_name, sess, graph, model, class_names, ))
      t.start()
      t.join()

@sio.on('test-request')
def send_test_msg():
  sio.emit('attend', json.dumps({
    'user_id': ['1']
  }))

def classify_process(sess_name, sess, graph, model, class_names):
  start = time.time()
  sess_dir = 'tmp/' + sess_name
  align_result = align_dataset_mtcnn(sess_name)
  if align_result == None:
    print('Face not detected!')
    shutil.rmtree(sess_dir)
    return
  print(align_result)
  tracking(align_result[0], align_result[1], sess_name)
  try:
    with graph.as_default():
      dataset = facenet.get_dataset(sess_dir + '_t')
      paths, labels = facenet.get_image_paths_and_labels(dataset)

      # Get input and output tensors
      images_placeholder = graph.get_tensor_by_name("input:0")
      embeddings = graph.get_tensor_by_name("embeddings:0")
      phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
      embedding_size = embeddings.get_shape()[1]

      # Preprocess images
      nrof_images = len(paths)
      nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
      emb_array = np.zeros((nrof_images, embedding_size))
      for i in range(nrof_batches_per_epoch):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
      
      predictions = model.predict_proba(emb_array)
      best_class_indices = np.argmax(predictions, axis=1)
      best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

      label_to_class_map = create_label_to_class_map(class_names)
      print(label_to_class_map)

      data_dir_result_label_map = create_data_dir_result_label_map(paths, best_class_indices, best_class_probabilities)

      data_dir_result_class_map = {}
      for dir_name in data_dir_result_label_map:
        label = data_dir_result_label_map[dir_name]
        data_dir_result_class_map[dir_name] = label_to_class_map[label]

      print(data_dir_result_class_map)
      end = time.time()
      print('Total time elapsed:', (end - start))

      processed_names = []
      for name in data_dir_result_class_map.values():
        if name == 'unknown' or name in processed_names:
          continue
        processed_names.append(name)
        try:
          update_attendance(name)
        except Exception as e:
          print(e)
        sio.emit('attend', name)
      
      for i in range(len(paths)):
        filename = paths[i].split('/')[-1]
        path = paths[i][:-len(filename)]
        acc = round(best_class_probabilities[i]*50,4)
        acc = pow(acc, 3)
        acc = math.sqrt(acc)
        acc = round(acc)
        if acc >= 200:
          for imgname in dataset:
            t = str(time.time())
            shutil.move(path + '/' + imgname, 'dataset/{}/{}.jpg'.format(label_to_class_map[best_class_indices[i]], t))

  except Exception as e:
    print(e)

if __name__ == '__main__':
  t = Thread(target=background_task, args=())
  t.start()
  app.run(threaded=True, port=5001)
