from align_dataset_mtcnn import *
from classifier import *
from handle_files import *
import shutil
from datetime import datetime
'''
공통적으로
detection folder안에
tracking을 한 같은 사람얼굴끼리 모아둔 폴더(face1 folder, face2 folder, face3 folder…..)
처음 등록 시:
이름을 받으면 각각에 해당하는 폴더 이름이 바뀜 -> 잘못된 사진이 있는지 수동으로 확인 -> train folder로 이동 => train
<train>
input: (train folder 안의) 폴더들 이름으로 labeling
output: train 결과와 라벨정보가 있는 pickle생성
<classify>
input: (detection folder안) face가 폴더 별로 정리되어있음
output: 각각의 폴더에서 예측한 것 중 비율이 제일 큰 것을 최종 결과값으로 하고 폴더 이름을 해당 이름으로 바꿈
——> 예측값이 40퍼 이하인 것은 지우고 train folder로 이동시켜 train 시킴
'''

# original data dir (before crop)
train_input_dir = # 'your train original data dir path'
classify_input_dir = # 'your classify original data dir path'

# after crop
# labeling by class
train_data_dir = # 'train data dir path'
classify_data_dir = # 'detection data dir path'


def facenet_main(request, train_data_dir, classify_data_dir, class_dir ='./class', classifier_filename = None, model_path = './pretrainedModel2017/20170512-110547.pb'):
	# get now time
	now = datetime.now()
	now = now.strftime('%y-%m-%d_%H:%M:%S')

	make_nonexisted_dir(class_dir)
	classifier_path = None
	classifier_file_list = remove_DS_Store_in_list(os.listdir(class_dir))

	# Assign to recent model if not specified	
	if classifier_filename == None:
		# Assign to classifier.pkl if there is no model 
		if not classifier_file_list:
			classifier_filename = 'classifier.pkl'
			classifier_path = os.path.join(class_dir, classifier_filename)
		# not empty folder
		else:
			# save model by now time
			if request == 'TRAIN':
				classifier_filename = 'classifier_' + now + '.pkl'
				classifier_path = os.path.join(class_dir, classifier_filename)

			elif request == 'CLASSIFY':
				classifier_path = get_latest_file_in_dir(class_dir)
				classifier_filename = classifier_path.split('/')[-1]

	# specified model
	else:
		classifier_path = os.path.join(class_dir, classifier_filename)


	print('classifier filename : %s' %classifier_filename)
	print()

	if request == 'TRAIN':
		classifier(train_data_dir, model_path, classifier_path, request)

	elif request == 'CLASSIFY':
		if not classifier_file_list:
			print('There is no model')
			return 0
		elif not classifier_filename in classifier_file_list:
			print('There is no such model : %s' %classifier_filename)
			return 0		
		else:
			classifier(classify_data_dir, model_path, classifier_path, request)
			# move_all_files_to_other_dir(classify_data_dir, train_data_dir)

	elif request == 'REGISTER_SUCCESS':
		#detection dir -> train dir
		move_all_files_to_other_dir(classify_data_dir, train_data_dir)
		
# # crop image
# align_dataset_mtcnn(train_input_dir, train_data_dir, image_size = 160)
# align_dataset_mtcnn(classify_input_dir, classify_data_dir, image_size = 160)


facenet_main('TRAIN', train_data_dir, classify_data_dir)
facenet_main('CLASSIFY', train_data_dir, classify_data_dir)


