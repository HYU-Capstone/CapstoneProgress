from face_detect import *
from face_tracking import *

input_dir = 'C:/Users/JB/Desktop/input'
output_dir = 'C:/Users/JB/Desktop/output'
bounding_boxes_filename = 'C:/Users/JB/Desktop/output/bounding_boxes_txt/bounding_boxes_0.txt'
saving_path = "C:/Users/JB/Desktop/face/"


align_dataset_mtcnn(input_dir, output_dir)
tracking(bounding_boxes_filename, saving_path, input_dir)
