import tensorflow as tf
import numpy as np
from facenet import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
import datetime 
from imblearn.combine import SMOTETomek
from files_manager_functions import *

data_dir = 'dataset'
model_path = 'models/20180402-114759.pb'
batch_size = 90
image_size = 160
today = datetime.datetime.now().strftime('%Y%m%d')
classifier_filename = 'models/{}.clf'.format(today)

min_nrof_images_per_class = 40
nrof_train_images_per_class = 20

def main():
  with tf.Graph().as_default() as _:
    with tf.Session() as sess:
      if get_num_of_files_in_dir(data_dir) == 0:
        print('No train data!')
        return
      
      dataset = facenet.get_dataset(data_dir)

      for cls in dataset:
        assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            
      paths, labels = facenet.get_image_paths_and_labels(dataset)
      
      print('Number of classes: %d' % len(dataset))
      print('Number of images: %d' % len(paths))
      
      # Load the model
      print('Loading feature extraction model')
      facenet.load_model(model_path)

      # Get input and output tensors
      images_placeholder = sess.graph.get_tensor_by_name("input:0")
      embeddings = sess.graph.get_tensor_by_name("embeddings:0")
      phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
      embedding_size = embeddings.get_shape()[1]
      
      # Run forward pass to calculate embeddings
      print('Calculating features for images')
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
      
      classifier_filename_exp = os.path.expanduser(classifier_filename)

      x, y = SMOTETomek(random_state=4).fit_sample(emb_array, labels)

      print('Training classifier')
      model = SVC(kernel='linear', probability=True)
      model.fit(x, y)
  
      # Create a list of class names
      class_names = [ cls.name.replace('_', ' ') for cls in dataset]

      # Saving classifier model
      with open(classifier_filename_exp, 'wb') as outfile:
          pickle.dump((model, class_names), outfile)
      print('Saved classifier model to file "%s"' % classifier_filename_exp)

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
  train_set = []
  test_set = []
  for cls in dataset:
    paths = cls.image_paths
    # Remove classes with less than min_nrof_images_per_class
    if len(paths) >= min_nrof_images_per_class:
      np.random.shuffle(paths)
      train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
      test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
  return train_set, test_set

if __name__ == "__main__":
  main()