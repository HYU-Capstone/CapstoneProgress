from facenet import facenet
import tensorflow as tf
import math
import os
import numpy as np
import pickle
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

data_dir = 'testset'
model_path = 'models/20180402-114759.pb'
batch_size = 90
image_size = 160
classifier_filename = 'models/20190523.clf'

dataset = facenet.get_dataset(data_dir)
paths, labels = facenet.get_image_paths_and_labels(dataset)
print('Number of classes: %d' % len(dataset))
print('Number of images: %d' % len(paths))

classifier_filename_exp = os.path.expanduser(classifier_filename)

with open(classifier_filename_exp, 'rb') as infile:
  (model, class_names) = pickle.load(infile)

print('Loaded classifier model from file "%s"' % classifier_filename_exp)

with tf.Graph().as_default():
  with tf.Session() as sess:
    
    for cls in dataset:
      assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            
      paths, labels = facenet.get_image_paths_and_labels([cls])

      # Load the model
      # print('Loading feature extraction model')
      facenet.load_model(model_path)

      # Get input and output tensors
      images_placeholder = sess.graph.get_tensor_by_name("input:0")
      embeddings = sess.graph.get_tensor_by_name("embeddings:0")
      phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
      embedding_size = embeddings.get_shape()[1]

      # Run forward pass to calculate embeddings
      # print('Calculating features for images')
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
      
      correct = 0
      total = 0
      for i in range(len(best_class_indices)):
        total += 1
        if class_names[best_class_indices[i]] == cls.name:
          correct += 1
        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
      
      accuracy = correct / total
      print('{} => Accuracy: %.3f'.format(cls.name) % accuracy)
