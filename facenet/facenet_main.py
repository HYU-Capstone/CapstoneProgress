from align_dataset_mtcnn import *
from classifier import *

# folder로 labeling됨
train_input_dir = ## crop 전 train dir
classify_input_dir = ## crop 전 classify dir

train_data_dir = ## crop 후 train_data
classify_data_dir = ## crop 후 classify dir

model_path = './pretrainedModel/2017/20170512-110547.pb'


# crop train data
align_dataset_mtcnn(train_input_dir, train_data_dir, image_size = 160)
align_dataset_mtcnn(classify_input_dir, classify_data_dir, image_size = 160)
# crop test data
classifier(train_data_dir, model_path, 'TRAIN')
classifier(classify_data_dir, model_path, 'CLASSIFY')
