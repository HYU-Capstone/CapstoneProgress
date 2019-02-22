from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from models import CNN
from manageData import get_n_classes, create_dataset, USAGE_MODE
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')
print(mnist.test.images.shape)
images = tf.Variable(mnist.test.images, name='images')

DATASET_PATH = os.getcwd() + '/TrainData'

TEST_DATASET_PATH = os.getcwd() + '/TestData'

LOG_DIR = os.getcwd() + '/tensorboard'
NAME_TO_VISUALISE_VARIABLE = 'embedding'

IMG_HEIGHT = 100
IMG_WIDTH = 100
CHANNELS = 3

LEARNING_RATE = 0.001
BATCH_SIZE = 400
DROPOUT = 0.70

MAP_FILE_NAME = 'map_categorical_classes.txt'

SAVE_PATH = 'model'
MODEL_NAME = 'CNN'

if not os.path.exists(SAVE_PATH):
	os.makedirs(SAVE_PATH)

if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)


# num_steps = 1
# display_step = 1

num_steps = 500
display_step = 1


full_save_path = os.path.join(SAVE_PATH, MODEL_NAME)

dataset = create_dataset(DATASET_PATH, IMG_HEIGHT, IMG_WIDTH, CHANNELS, MAP_FILE_NAME, USAGE_MODE[0], BATCH_SIZE)
N_CLASSES = get_n_classes()
print(N_CLASSES)
test_dataset = create_dataset(TEST_DATASET_PATH, IMG_HEIGHT, IMG_WIDTH, CHANNELS, MAP_FILE_NAME, USAGE_MODE[1])


iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

iterator2 = test_dataset.make_initializable_iterator()
testX, testY = iterator2.get_next()

## false로 했을때 value error

logits_train = CNN(X, N_CLASSES, dropout_rate=DROPOUT, reuse=False, is_training=True)
logits_test = CNN(testX, N_CLASSES)

cnn_net_name = 'ConvNet2'
logits_train = logits_train.conv_net2()
logits_test = logits_test.conv_net2()

regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, cnn_net_name)
for weight in regularization:
	print(weight)
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits_train, labels=Y))
loss_op = cross_entropy + tf.reduce_sum(regularization)

#momentum / adaGrad / RMSProp / adam
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)

prediction = tf.argmax(logits_test, 1, name='prediction')
correct_pred = tf.equal(prediction, tf.cast(testY, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

tensorboard_saver = tf.train.Saver([images])


num = 0
setting_value_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
acc_dict = {}
past_acc = 0
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(iterator.initializer)
	sess.run(iterator2.initializer)
	print(logits_train)
	for step in range(1, num_steps+1):
		print(step)
		acc = sess.run(accuracy)
		if acc >= setting_value_list[0]:
			for setting_value in setting_value_list:
				if acc >= setting_value and acc < setting_value+0.5:
					saver_path = saver.save(sess, full_save_path, num)
					num += 1
					setting_value_list.remove(setting_value)
					acc_dict[saver_path] = acc
					break
		
		#best
		if acc > past_acc:
			best_saver_path = saver.save(sess, full_save_path)
			acc_dict[best_saver_path] = acc
			past_acc = acc

		if step % display_step == 0:
			_, loss, correct = sess.run([train_op, loss_op, correct_pred])
			print("Step " + str(step) + \
				", Minibatch Loss= {:.4f}".format(loss) + \
				", Training Accuracy= {:.3f}".format(acc))
		else:
			sess.run(train_op)
	tensorboard_saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))

	print("Optimization Finished!")
	print(acc_dict)
	saver.save(sess, full_save_path, num)
	print(best_saver_path)
