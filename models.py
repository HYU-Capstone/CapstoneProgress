import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, dense, dropout
from tensorflow.contrib.layers import flatten, xavier_initializer, l2_regularizer
from tensorflow.nn import softmax, relu

		


def conv(inputs, filters, kernel_size, regularization_scale, name):
	initializer = xavier_initializer()
	regularizer = l2_regularizer(scale=regularization_scale)
	_conv = conv2d(inputs, filters, kernel_size, padding='SAME', activation=relu, name=name, kernel_initializer=initializer, kernel_regularizer=regularizer)
	return _conv

def conv_layer(inputs, filters, kernel_size, name, regularization_scale):
	_conv = conv(inputs, filters, kernel_size, name, regularization_scale)
	_conv = max_pooling2d(_conv, 2, 2 , padding='SAME')
	return _conv	

def affine_layer(inputs, units, regularization_scale, name):
	initializer = xavier_initializer()	
	regularizer = l2_regularizer(scale=regularization_scale)
	return dense(inputs, units, activation=relu, name=name, kernel_initializer = initializer, kernel_regularizer=regularizer)

def out_layer(inputs, n_class, is_training, regularization_scale, name):
	initializer = xavier_initializer()
	regularizer = l2_regularizer(scale=regularization_scale)
	out = dense(inputs, n_class, kernel_initializer = initializer, kernel_regularizer=regularizer, name=name)
	out = softmax(out) if not is_training else out
	return out


class CNN:
	def __init__(self, data, n_class, dropout_rate=None, reuse=True, is_training=False, regularization_scale=0.001):
		self.data = data
		self.dropout_rate = dropout_rate
		self.n_class = n_class
		self.reuse = reuse
		self.is_training = is_training
		self.regularization_scale = regularization_scale

	def conv_net(self):
		with tf.variable_scope('ConvNet', reuse=self.reuse):
			conv1 = conv_layer(self.data, 32, 3, self.regularization_scale, 'conv1')
			conv2 = conv_layer(conv1, 64, 3, self.regularization_scale, 'conv2')
			
			affine = flatten(conv2)
			affine = affine_layer(affine, 1024, self.regularization_scale, 'affine')
			affine = dropout(affine, rate=self.dropout_rate, training=self.is_training)
			
			out = out_layer(affine, self.n_class, self.is_training, self.regularization_scale, 'out')
		
		return out

	def conv_net2(self):
		with tf.variable_scope('ConvNet2', reuse=self.reuse):
				conv1 = conv_layer(self.data, 16, 3, self.regularization_scale, 'conv1')
				conv2 = conv_layer(conv1, 32, 3, self.regularization_scale, 'conv2')
				conv3 = conv_layer(conv2, 64, 3, self.regularization_scale, 'conv3')
				conv4 = conv_layer(conv3, 128, 3, self.regularization_scale, 'conv4')
			
				affine = flatten(conv4)
				affine1 = affine_layer(affine, 512, self.regularization_scale, 'affine1')
				affine2 = affine_layer(affine1, 1024, self.regularization_scale, 'affine2')
				affine = dropout(affine2, rate=self.dropout_rate, training=self.is_training)
			
				out = out_layer(affine, self.n_class, self.is_training, self.regularization_scale, 'out')
		
		return out
	#VGG16
	def VGG_net(self):
		with tf.variable_scope('VGGNet', reuse=self.reuse):
			conv1_1 = conv(self.data, 64, 3, self.regularization_scale, 'conv1_1')
			conv1_2 = conv(conv1_1, 64, 3, self.regularization_scale, 'conv1_2')
			pool1 = max_pooling2d(conv1_2, 2, 2)

			conv2_1 = conv(pool1, 128, 3, self.regularization_scale, 'conv2_1')
			conv2_2 = conv(conv2_1, 128, 3, self.regularization_scale, 'conv2_2')
			pool2 = max_pooling2d(conv2_2, 2, 2)

			conv3_1 = conv(pool2, 256, 3,  self.regularization_scale, 'conv3_1')
			conv3_2 = conv(conv3_1, 256, 3,  self.regularization_scale, 'conv3_2')
			conv3_3 = conv(conv3_2, 256, 3, self.regularization_scale, 'conv3_3')
			pool3 = max_pooling2d(conv3_3, 2, 2)

			conv4_1 = conv(pool3, 512, 3, self.regularization_scale, 'conv4_1')
			conv4_2 = conv(conv4_1, 512, 3, self.regularization_scale, 'conv4_2')
			conv4_3 = conv(conv4_2, 512, 3, self.regularization_scale, 'conv4_3')
			pool4 = max_pooling2d(conv4_3, 2, 2)

			conv5_1 = conv(pool4, 512, 3, self.regularization_scale, 'conv5_1')
			conv5_2 = conv(conv5_1, 512, 3, self.regularization_scale, 'conv5_2')
			conv5_3 = conv(conv5_2, 512, 3, self.regularization_scale, 'conv5_3')
			pool5 = max_pooling2d(conv5_3, 2, 2)

			affine = flatten(pool5)
			affine_1 = affine_layer(affine, 4096, self.regularization_scale, 'affine_1')
			affine_1 = dropout(affine_1, rate=self.dropout_rate, training=self.is_training)
			affine_2 = affine_layer(affine_1, 4096, self.regularization_scale, 'affine_2')
			affine_2 = dropout(affine_2, rate=self.dropout_rate, training=self.is_training)

			out = out_layer(affine_2, self.n_class, self.is_training, self.regularization_scale, 'out')
			
		return out
