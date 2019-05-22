from face_detect import *
from face_tracking import *
from facenet_main import facenet_train, facenet_classify
input_dir = #
train_data_dir = #
classify_data_dir = #

bounding_boxes_filename = #



align_dataset_mtcnn(input_dir, classify_data_dir)
tracking(bounding_boxes_filename, input_dir, classify_data_dir)
facenet_classify(train_data_dir, classify_data_dir)