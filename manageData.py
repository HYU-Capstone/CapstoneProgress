import os
import tensorflow as tf
import cv2
import numpy as np


IMG_HEIGHT = 0
IMG_WIDTH = 0
CHANNELS = 0

_N_CLASSES = 1

MAP_MODE = ['label_to_class', 'class_to_label']
USAGE_MODE = ['train', 'test']

IMAGEPATHS = None

def _create_map_file(file_name, classes, class_to_label_dict):
	with open(file_name,'w') as map_file:
		for _class in classes:
			label = class_to_label_dict[_class]
			map_file.write(str(label)+" "+_class+" "+os.linesep)


def _read_image(path):
	path = str(path, 'utf-8')
	return cv2.imread(path, cv2.IMREAD_COLOR)

def _read_data(path, label):
	image = _read_image(path)
	
	#make gray
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #   image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		# cv2.THRESH_BINARY_INV,11,2)
	# image = np.expand_dims(image, -1)

	label = np.array(label, dtype=np.int64)
	image = image.astype(np.int32)
	return image, label
	
def _resize_data(image_decoded, label):
	image_decoded.set_shape([None, None, CHANNELS])
	image_resized = tf.image.resize_images(image_decoded, [IMG_HEIGHT,IMG_WIDTH])
	return image_resized, label
def _set_image_option(image, label):
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_brightness(image,max_delta=0.5)
	image = tf.image.random_contrast(image,lower=0.2,upper=2.0)
	image = tf.image.random_hue(image,max_delta=0.08)
	image = tf.image.random_saturation(image,lower=0.2,upper=2.0)
	return image, label

def _data_normalization(image, label):
	image = tf.image.per_image_standardization(image)
	return image, label

def create_map_dict(file_name, mode):
	map_dict = {}
	with open(file_name,'r') as map_file:
			lines = map_file.readlines()
			for line in lines:
				split = line.split(' ')
				label = int(split[0])
				_class = split[1]
				if mode == 'label_to_class':
					map_dict[label] = _class
				elif mode == 'class_to_label':
					map_dict[_class] = label
				else :
					raise Exception("Unknown mode")

	return map_dict



def create_dataset(dataset_path, img_height, img_width, channels, map_file_name, usage, batch_size = None):
	imagepaths, labels, classes = list(), list(), list()
	class_to_label_dict = {}
	
	global IMG_HEIGHT, IMG_WIDTH, CHANNELS, _N_CLASSES, IMAGEPATHS
	IMG_HEIGHT, IMG_WIDTH, CHANNELS = img_height, img_width, channels

	label = 0
	files = os.listdir(dataset_path)
	if usage == USAGE_MODE[1]:
		class_to_label_dict = create_map_dict(map_file_name, MAP_MODE[1])

	if '.DS_Store' in files:
		files.remove('.DS_Store')
	for file in files:
		_class = file.split('_')[0]
		if usage == USAGE_MODE[0]:
			if not _class in classes:
				classes.append(_class)
				class_to_label_dict[_class] = label
				label += 1
		
		if file.endswith('.jpg'):
			imagepaths.append(os.path.join(dataset_path, file))
			labels.append(class_to_label_dict[_class])
	
	n_files = len(imagepaths)
	IMAGEPATHS = imagepaths
	print(class_to_label_dict)
	_N_CLASSES = len(classes)

	if batch_size == None:
		batch_size = n_files

	if usage == USAGE_MODE[0]:
		_create_map_file(map_file_name, classes, class_to_label_dict)

	dataset = tf.data.Dataset.from_tensor_slices((imagepaths, labels))
	dataset = dataset.map(lambda images, labels:
						 tuple(tf.py_func(_read_data, [images, labels], [tf.int32, tf.int64])))
	
	dataset = dataset.map(_resize_data)
	dataset = dataset.map(_set_image_option)
	dataset = dataset.map(_data_normalization)
	dataset = dataset.repeat()
	if usage == USAGE_MODE[0]:
		dataset = dataset.shuffle(buffer_size=(int(n_files * 0.4) + 3 * batch_size))
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_initializable_iterator()
	image_stacked, label_stacked = iterator.get_next()
	next_element = iterator.get_next()

	return dataset

def get_n_classes():
	return _N_CLASSES

def get_imagepaths():
	return IMAGEPATHS

