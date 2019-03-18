"""
<modified>
An example of how to use your own dataset to train a classifier that recognizes people.
"""
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

## train한 것 세이브하고 그 세이브와 트레인한 데이터 모아두기(구분) / 자동화
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import facenet
import os
import sys
import tensorflow as tf
import numpy as np
import pickle
import math
from sklearn.svm import SVC

import shutil

def create_label_to_class_map(class_to_label_map):
	class_list = list(class_to_label_map.keys())
	label_to_class_map = {}
	for i in range(len(class_list)):
		class_name = class_list[i]
		label_name = class_to_label_map[class_name]
		label_to_class_map[int(label_name)] = class_name
	return label_to_class_map

def create_wrong_imagepath_to_label_map(best_class_indices, paths, labels, class_to_label_map):
	label_to_class_map = create_label_to_class_map(class_to_label_map)
	
	pred_matched_list= list()
	for i in range(len(best_class_indices)):
		if best_class_indices[i] == labels[i]:
			pred_matched_list.append(True)
		else:
			pred_matched_list.append(False)

	false_list = [i for i, x in enumerate(pred_matched_list) if not x]

	wrong_imagepath_to_label_map = {}
	for i in range(len(false_list)):
		index = false_list[i]
		imagepath = paths[index]
		prediction_label = best_class_indices[index]
		correct_label = labels[index]
		wrong_imagepath_to_label_map[imagepath] = label_to_class_map[prediction_label]

	return wrong_imagepath_to_label_map


def copy_wrong_recognition_files_to_dir(wrong_imagepath_to_label_map, copy_path='./wrongRecogntion'):
	make_nonexisted_dir(copy_path)
	wrong_imagepath_list = list(wrong_imagepath_to_label_map.keys())

	for wrong_imagepath in wrong_imagepath_list:
		shutil.copy(wrong_imagepath, copy_path)
		filename = wrong_imagepath.split('/')[-1]
		correct_label = wrong_imagepath.split('/')[-2]
		wrong_label = wrong_imagepath_to_label_map[wrong_imagepath]
		copied_file_path = os.path.join(copy_path, filename)

		num = 0
		rename = "recognize "+correct_label+" as "+wrong_label+"_"+str(num)+".jpg"
		rename_path = os.path.join(copy_path, rename)
		while os.path.exists(rename_path):
			num += 1
			rename = "recognize "+correct_label+" as "+wrong_label+"_"+str(num)+".jpg"
			rename_path = os.path.join(copy_path, rename)
		os.rename(copied_file_path, rename_path)


def print_wrong_recognition_file_details(wrong_imagepath_to_label_map):
	## Make labels by filename (filename form : personname + '_' + #)
	wrong_recognition_file_path = list(wrong_imagepath_to_label_map.keys())

	print('imagepath : \tprediction\tcorrect')
	for file in wrong_recognition_file_path:
		person_name = file.split('/')[-2]
		print(file + ' : \t' + wrong_imagepath_to_label_map[file] + '\t' + person_name + '\n' + '---------------------------')

def print_details_of_all_recognition_result(best_class_indices, best_class_probabilities, class_to_label_map):
	# best class, best probability per image
	class_names = list(class_to_label_map.keys())
	for i in range(len(best_class_indices)):
		print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

def print_accuracy(best_class_indices, paths, labels, class_to_label_map):
	accuracy = np.mean(np.equal(best_class_indices, labels))
	print('Accuracy: %.3f' % accuracy)


def classifier(data_dir, model_path,  mode, class_dir='./class', classifier_filename='classifier.pkl', batch_size=1000, image_size=160):
	paths = None
	data_dir_names = os.walk(data_dir).__next__()[1]
	if 'bounding_boxes_txt' in data_dir_names:
		data_dir_names.remove('bounding_boxes_txt')
	if data_dir_names == []:
		print('There is no train data')
		return 0
	dataset = facenet.get_dataset(data_dir)
	paths, labels = facenet.get_image_paths_and_labels(dataset)
	
	with tf.Graph().as_default():
		with tf.Session() as sess:
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
			
			classifier_path = os.path.join(class_dir, classifier_filename)

			# Train classifier	
			if (mode=='TRAIN'):
				print('Training classifier')
				model = SVC(kernel='linear', probability=True)
				model.fit(emb_array, labels)
			
				# Create a list of class names
				class_to_label_map = {}
				label = 0
				for cls in dataset:
					person_name = cls.name.replace('_', ' ')
					class_to_label_map[person_name] = label
					label += 1

				if not os.path.exists(class_dir):
					os.makedirs(class_dir)
				
				# Saving classifier model, class-label dict
				with open(classifier_path, 'wb') as outfile:
					pickle.dump((model, class_to_label_map), outfile)
				print('Saved classifier model to file "%s"' % classifier_path)
				
			# Classify images
			elif (mode=='CLASSIFY'):
				##todo: distinct test vs practice 
				print('Testing classifier')
				# Load classifier model, class-label dict
				with open(classifier_path, 'rb') as infile:
					(model, class_to_label_map) = pickle.load(infile)

				print('Loaded classifier model from file "%s"' % classifier_path)

				predictions = model.predict_proba(emb_array)
				best_class_indices = np.argmax(predictions, axis=1)
				best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
				

				label_to_class_map = create_label_to_class_map(class_to_label_map)
				wrong_imagepath_to_label_map = create_wrong_imagepath_to_label_map(best_class_indices, paths, labels, class_to_label_map)

				copy_wrong_recognition_files_to_dir(wrong_imagepath_to_label_map)
				print_wrong_recognition_file_details(wrong_imagepath_to_label_map)
				print_details_of_all_recognition_result(best_class_indices, best_class_probabilities, class_to_label_map)
				print_accuracy(best_class_indices, paths, labels, class_to_label_map)

def test_accuracy_different_depending_on_size(train_mode, train_data_path, train_data_resize_path_name, test_mode, test_data_path, test_data_resize_path_name, train_data_dir, classify_data_dir, classifier_filename):
	#image size 140~170
	image_size = 140
	while 140 <= image_size and image_size <= 170:
		i = image_size
		print("original image_size : " + str(image_size))
		while 140 <= i:
			print("    image_size : " + str(i))
			resize_images(image_size,train_data_path,train_data_resize_path_name, train_mode)
			resize_images(image_size,test_data_path,test_data_resize_path_name, test_mode)
			classifier(train_data_dir, model_path, 'TRAIN', classifier_filename=classifier_filename, image_size=i)
			classifier(classify_data_dir, model_path, 'CLASSIFY', classifier_filename=classifier_filename, image_size=i)
			i -= 10
		image_size += 10

