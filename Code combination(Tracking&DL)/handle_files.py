import os
import shutil

def make_nonexisted_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

def remove_DS_Store_in_list(file_list):
	if '.DS_Store' in file_list:
		file_list.remove('.DS_Store')
	return file_list

def remove_filename_in_list(file_list, filename):
	if filename in file_list:
		file_list.remove(filename)
	return file_list

def remove_all_files_in_dir(dir):
	for root, dirs, files in os.walk(dir):
		for f in files:
			os.unlink(os.path.join(root, f))
		for d in dirs:
			shutil.rmtree(os.path.join(root, d))
	shutil.rmtree(dir)
	
def remove_file_in_dir(dir, filename):
	file_path = os.path.join(dir, filename)
	if os.path.exists(file_path):
		os.remove(file_path)

def move_all_files_to_other_dir(current_dir, other_dir):
	dir_name_list = os.walk(current_dir).__next__()[1]
	for dir_name in dir_name_list:
		dir_path_in_other_dir = os.path.join(other_dir, dir_name)
		make_nonexisted_dir(dir_path_in_other_dir)

		dir_path_in_current_dir = os.path.join(current_dir, dir_name)
		files_per_dir = os.walk(dir_path_in_current_dir).__next__()[2]
		files_per_dir = remove_DS_Store_in_list(files_per_dir)
		for file in files_per_dir:
			file_path = os.path.join(dir_path_in_current_dir, file)
			try:
				shutil.move(file_path, dir_path_in_other_dir)
			except Exception as e:
				print(e)
		shutil.rmtree(dir_path_in_current_dir)
	print('move success')


def get_flie_list_in_dir(dir):
	file_list = os.listdir(dir)
	file_list = remove_DS_Store_in_list(file_list)
	return file_list

def get_latest_file_in_dir(dir):
	file_list = get_flie_list_in_dir(dir)

	file_path_list = list()
	for file in file_list:
		file_path = os.path.join(dir, file)
		file_path_list.append(file_path)
	latest_file = max(file_path_list, key=os.path.getctime)
	return latest_file

def get_num_of_files_in_dir(dir):
	file_list = get_flie_list_in_dir(dir)
	return len(file_list)

def rename_dir_name(root_dir_path, dir_name, rename):
	current_path = os.path.join(root_dir_path, dir_name)
	rename_path = os.path.join(root_dir_path, rename)
	num = 0
	while os.path.exists(rename_path):
		num += 1
		dup_rename = rename + str(num)
		rename_path = os.path.join(root_dir_path, dup_rename)
	shutil.move(current_path, rename_path)

def rename_filename(dir_path, filename, rename, extension='jpg'):
	current_path = os.path.join(dir_path, filename)
	rename_path = os.path.join(dir_path, rename + '.' + extension)
	num = 0
	while os.path.exists(rename_path):
		num += 1
		dup_rename = rename + '_' + str(num) + '.' + extension
		rename_path = os.path.join(dir_path, dup_rename)
	try:
		os.rename(current_path, rename_path)
	except Exception as e:
		print("error : " )
		print(e)
