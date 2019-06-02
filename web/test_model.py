# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from facenet import facenet
import os
import sys
import tensorflow as tf
import numpy as np
import pickle
import math
from sklearn.svm import SVC

import shutil
from files_manager_functions import *
from collections import defaultdict
from imblearn.combine import *

data_dir = 'testset'
# model_path = 'model/20170512-110547.pb'
model_path = 'models/20180402-114759.pb'

batch_size = 90
image_size = 160
classifier_path = 'models/20190530.clf'


def create_data_dir_result_label_map(paths, best_class_indices, best_class_probabilities, unknown_threshold=100, remove_threshold=180):
  dir_name = paths[0].split('/')[-2]
  data_dir_result_label_map = {}
  #the number of probabilities satisfying the criterion
  predict_label_satisfied_num_map = defaultdict(int)
  for i in range(len(paths)):
    cur_dir_name = paths[i].split('/')[-2]
    # initialize each time dir changes
    if dir_name != cur_dir_name or i == len(paths)-1:
      if len(predict_label_satisfied_num_map) == 0:
        data_dir_result_label_map[dir_name] = -1
      else:
        (key, value) = max(predict_label_satisfied_num_map.items(), key=lambda a: a[1])
        satisfied_num_of_key = predict_label_satisfied_num_map[key]
      # set 'unknown' if failure to pass the criteria
      if satisfied_num_of_key < 2:
        data_dir_result_label_map[dir_name] = -1
      else :
        data_dir_result_label_map[dir_name] = key
      predict_label_satisfied_num_map = defaultdict(int)        
      dir_name = cur_dir_name

    # run only when passed criteria
    acc = round(best_class_probabilities[i]*50,4)
    acc = pow(acc, 3)
    acc = math.sqrt(acc)
    acc = round(acc)
    if acc >= unknown_threshold:
      predict_label = best_class_indices[i]
      predict_label_satisfied_num_map[predict_label] += 1
    # failure to pass the criteria
    # if remove_threshold > acc:
    #     os.remove(paths[i])

  return data_dir_result_label_map

def create_label_to_class_map(class_names):
  label_to_class_map = {}
  for i in range(len(class_names)):
    label_to_class_map[i] = class_names[i]
  label_to_class_map[-1] = 'unknown'
  return label_to_class_map

def rename_dirs_to_result_label(root_dir_path, data_dir_names, data_dir_result_label_map, label_to_class_map):
  #만약 똑같은 dir있으면_1
  for data_dir_name in data_dir_names:
    label = data_dir_result_label_map[data_dir_name]
    _class = label_to_class_map[label]
    rename_dir_name(root_dir_path, data_dir_name, _class)
    # shutil.move(os.path.join(root_dir_path, data_dir_name), os.path.join(root_dir_path, _class))


def rename_all_files_by_details(best_class_indices, best_class_probabilities, paths, label_to_class_map):
  for i in range(len(paths)):
    filename = paths[i].split('/')[-1]
    path = paths[i][:-len(filename)]
    acc = round(best_class_probabilities[i]*50,4)
    acc = pow(acc, 3)
    acc = math.sqrt(acc)
    acc = round(acc)
    rename = label_to_class_map[best_class_indices[i]] + '({}, {})'.format(acc, best_class_probabilities[i])
    rename_filename(path, filename, rename)
  print('end')

def print_details_of_all_recognition_result(best_class_indices, best_class_probabilities, class_to_label_map):
  # best class, best probability per image
  class_names = list(class_to_label_map.keys())
  for i in range(len(best_class_indices)):
    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

def classifier(data_dir, model_path, classifier_path):
  with tf.Graph().as_default():
    with tf.Session() as sess:
      if (get_num_of_files_in_dir(data_dir) == 0):
        print('There is no train data')
        return 0

      dataset = facenet.get_dataset(data_dir)
      paths, labels = facenet.get_image_paths_and_labels(dataset)
      # Load pretrained model's train parameter
      facenet.load_model(model_path)
      images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
      embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
      phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
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
      
      # Load classifier model, class-label dict
      with open(classifier_path, 'rb') as infile:
        (model, class_names) = pickle.load(infile)
      print(class_names)
      # print('Loaded classifier model from file "%s"' % classifier_path)

      predictions = model.predict_proba(emb_array)
      # print(len(paths))
      # print(prediction_time_elapsed/len(paths))

      best_class_indices = np.argmax(predictions, axis=1)
      best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

      label_to_class_map = create_label_to_class_map(class_names)
      print(label_to_class_map)

      data_dir_result_label_map = create_data_dir_result_label_map(paths, best_class_indices, best_class_probabilities)


      data_dir_result_class_map = {}
      for dir_name in data_dir_result_label_map:
        label = data_dir_result_label_map[dir_name]
        data_dir_result_class_map[dir_name] = label_to_class_map[label]
        print('dir_name : '+ dir_name +'\t recognition : '+ label_to_class_map[label])

      rename_all_files_by_details(best_class_indices, best_class_probabilities, paths, label_to_class_map)
      return data_dir_result_class_map

if __name__ == "__main__":
  print(classifier(data_dir, model_path, classifier_path))
