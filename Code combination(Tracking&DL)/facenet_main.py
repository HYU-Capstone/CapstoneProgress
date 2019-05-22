from classifier import *
from handle_files import *
import shutil
from datetime import datetime
from time import clock

def determine_train_classifier_path(class_dir, classifier_filename):
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
			classifier_filename = 'classifier_' + now + '.pkl'
			classifier_path = os.path.join(class_dir, classifier_filename)
	# specified model
	else:
		classifier_path = os.path.join(class_dir, classifier_filename)


	print('classifier filename : %s' %classifier_filename)
	print()
	return classifier_path

def determine_classify_classifier_path(class_dir, classifier_filename):
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
			classifier_path = get_latest_file_in_dir(class_dir)
			classifier_filename = classifier_path.split('/')[-1]
	# specified model
	else:
		classifier_path = os.path.join(class_dir, classifier_filename)


	print('classifier filename : %s' %classifier_filename)
	print()
	if not classifier_file_list:
		print('There is no model')
		return 0
	elif not classifier_filename in classifier_file_list:
		print('There is no such model : %s' %classifier_filename)
		return 0
	else:	
		return classifier_path

def facenet_train(train_data_dir, classify_data_dir, class_dir ='./class', classifier_filename = None, model_path = './pretrainedModel2017/20170512-110547.pb'):
	classifier_path = determine_train_classifier_path(class_dir, classifier_filename)
	classifier(train_data_dir, model_path, classifier_path, 'TRAIN')

def facenet_classify(train_data_dir, classify_data_dir, class_dir ='./class', classifier_filename = None, model_path = './pretrainedModel2017/20170512-110547.pb'):
	classifier_path = determine_classify_classifier_path(class_dir, classifier_filename)
	if classifier_path == 0:
		return 0
	else:
		classifier(classify_data_dir, model_path, classifier_path, 'CLASSIFY')
		# move_all_files_to_other_dir(classify_data_dir, train_data_dir)
