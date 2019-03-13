import cv2
import tensorflow as tf
from tensorflow.nn import conv2d, bias_add, relu, max_pool
from tensorflow.contrib.layers import flatten, xavier_initializer
import os
import numpy as np
from manageData import get_n_classes, create_dataset, create_map_dict, MAP_MODE, USAGE_MODE, get_imagepaths

import shutil


CHANNELS = 3
IMG_HEIGHT = 100
IMG_WIDTH = 100
save_path = 'model'
recent_ckpt_path = tf.train.latest_checkpoint(save_path)


meta_graph = recent_ckpt_path + '.meta'

MAP_FILE_NAME = "map_categorical_classes.txt"



DATASET_PATH = os.getcwd() + '/TestData'

tf.reset_default_graph()


dataset = create_dataset(DATASET_PATH, IMG_HEIGHT, IMG_WIDTH, CHANNELS, MAP_FILE_NAME, USAGE_MODE[1])

def get_np_kernel_bias_by_name(sess, kernel_name, bias_name):
	kernel = sess.graph.get_tensor_by_name(kernel_name)
	bias = sess.graph.get_tensor_by_name(bias_name)
	kernel = kernel.eval()
	bias = bias.eval()

	return kernel, bias

def create_kernel_bias_list(sess,model_name,layer_name_list):
	kernel_list, bias_list = list(), list()
	for layer_name in layer_name_list:
		kernel_name = model_name+'/'+layer_name+'/kernel:0'
		bias_name = model_name+'/'+layer_name+'/bias:0'
		kernel, bias = get_np_kernel_bias_by_name(sess, kernel_name, bias_name)
		kernel_list.append(kernel)
		bias_list.append(bias)
	return kernel_list, bias_list

def show_usable_operations():
	for op in tf.get_default_graph().get_operations():
		print(op.name)

def conv(inputs, filters, bias):
	_conv = conv2d(inputs, filters, [1,1,1,1], padding='SAME')
	_conv = bias_add(_conv, bias)
	_conv = relu(_conv)
	return _conv

def conv_layer(inputs, filters, bias):
	_conv = conv(inputs, filters, bias)
	_conv = max_pool(_conv, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
	return _conv

def affine_layer(inputs, filters, bias):
	affine = tf.matmul(inputs, filters) + bias
	affine = relu(affine)
	return affine

def out_layer(inputs, filters, bias):
	return tf.matmul(inputs, filters)+bias

def conv_net(x, kernel_list, bias_list):
	kernel1, kernel2, kernel3, kernel4 = kernel_list
	bias1, bias2, bias3, bias4 = bias_list    

	conv1 = conv_layer(x, kernel1, bias1)
	conv2 = conv_layer(conv1, kernel2, bias2)

	affine = flatten(conv2)
	affine = affine_layer(affine, kernel3, bias3)

	out = out_layer(affine, kernel4, bias4)

	return out

def conv_net2(x, kernel_list, bias_list):
	kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7 = kernel_list
	bias1, bias2, bias3, bias4, bias5, bias6, bias7 = bias_list 
	
	conv1 = conv_layer(x, kernel1, bias1)
	conv2 = conv_layer(conv1, kernel2, bias2)
	conv3 = conv_layer(conv2, kernel3, bias3)
	conv4 = conv_layer(conv3, kernel4, bias4)

	affine = flatten(conv4)
	affine1 = affine_layer(affine, kernel5, bias5)
	affine2 = affine_layer(affine1, kernel6, bias6)

	out = out_layer(affine2, kernel7, bias7)

	return out


iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(iterator.initializer)

	saver = tf.train.import_meta_graph(meta_graph)
	saver.restore(sess, recent_ckpt_path)

	model_name = 'ConvNet2'
	layer_name_list = ['conv1','conv2', 'conv3', 'conv4','affine1', 'affine2', 'out']	

	kernel_list, bias_list = create_kernel_bias_list(sess,model_name,layer_name_list)
	logits = conv_net2(X, kernel_list, bias_list)
	predictions = tf.argmax(logits, 1)

	pred = sess.run(predictions)
	y= sess.run(Y)
	print(pred)
	print(len(pred))

	print(y)
	print(len(y))

	cor_pred = list()
	k = 0
	for i in range(len(pred)):
		if pred[i] == y[i]:
			k += 1
			cor_pred.append(True)
		else:
			cor_pred.append(False)
	imagepaths = get_imagepaths()
	label_to_class_dict = create_map_dict(MAP_FILE_NAME, MAP_MODE[0])
	false_list=list()
	for adsf in range(len(cor_pred)):
		if cor_pred[adsf] == False:
			false_list.append(adsf)
	imagepath_wrong_label_dict = {}
	false_image_copy_path = '/Users/NamHyunsil/Documents/MLtest/false_image'
	asdf_path = '/Users/NamHyunsil/Documents/MLtest/asdf'
	

	if not os.path.exists(false_image_copy_path):
		os.makedirs(false_image_copy_path)

	if not os.path.exists(asdf_path):
		os.makedirs(asdf_path)
	y = sess.run(Y)
	asdf_list = list()
	for i in range(len(false_list)):
		index = false_list[i]
		print(index)
		imagepath = imagepaths[index]
		imagepath_wrong_label_dict[imagepath] = label_to_class_dict[pred[index]]
		print((imagepaths[index]) + '\t'+str(pred[index])+'\t'+ str(y[index])+'\t' + label_to_class_dict[pred[index]] + '\n' + '---------------------------')

	acc =  k / len(cor_pred)
	print(acc)

