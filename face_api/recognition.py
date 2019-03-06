import cognitive_face as CF
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import os
from random import shuffle
import time

KEY = '9812c727bb6a45e7959bac9e5c44cdf2'
BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0' 
PERSON_GROUP_ID = 'known-persons'
TRAIN_SET_PATH = os.path.join(os.getcwd(), 'TrainData')
TEST_SET_PATH = os.path.join(os.getcwd(), 'TestData')

BATCH_SIZE_PER_PERSON = 20

CF.Key.set(KEY)
CF.BaseUrl.set(BASE_URL)

STEPS = 400



def hanldle_error_generally(cf_func, *args):
	while True:
		try:
			result = cf_func(*args)
			return result
		except CF.CognitiveFaceException as e:
			# print e.code
			if e.code == 'RateLimitExceeded':
				time.sleep(1)
			if e.code == 'PersonGroupNotTrained':
				return e.code

def get_person_group_id_list(person_group_list):
	person_group_id_list = list()
	for person_group in person_group_list:
		person_group_id = person_group['personGroupId']
		person_group_id_list.append(person_group_id)
	return person_group_id_list

def create_person_group(person_group_id_list):
	if PERSON_GROUP_ID in person_group_id_list:
		hanldle_error_generally(CF.person_group.delete, PERSON_GROUP_ID)
		person_group_id_list.remove(PERSON_GROUP_ID)
	hanldle_error_generally(CF.person_group.create, PERSON_GROUP_ID)
	person_group_id_list.append(PERSON_GROUP_ID)
	print person_group_id_list

def detect_face(img_url, training=False):
	detection_faces = list()
	faces = hanldle_error_generally(CF.face.detect, img_url)
	print faces

	face_ids = [face['faceId'] for face in faces]
	identified_faces = hanldle_error_generally(CF.face.identify, face_ids, PERSON_GROUP_ID)
	if identified_faces == 'PersonGroupNotTrained':
		return identified_faces
	else:
		for identified_face in identified_faces:
			face_candidates = identified_face['candidates']
			if face_candidates==[]:
				detection_faces.append(None)
			else:
				best_candidate = None
				for face_candidate in face_candidates:
					print face_candidate['confidence']
					if face_candidate['confidence'] == 1 and training == True:
						print 'same data'
						# same data
						return 'SameData'
					#just for confirmation
					elif face_candidate['confidence'] == 1 and training == False:
						print 'same data'

					if best_candidate == None:
						best_candidate = face_candidate
					elif face_candidate['confidence'] > best_candidate['confidence']:
						best_candidate = face_candidate
				matched_person = hanldle_error_generally(CF.person.get, PERSON_GROUP_ID, best_candidate['personId'])
				name = matched_person['name']
				detection_faces.append(name)
		return detection_faces


def add_face_for_batch_size(name, person_data_list, batch_data_list):
	shuffle(person_data_list)
	person_num = 0
	index = 0
	while person_num != BATCH_SIZE_PER_PERSON and index != len(person_data_list):
		img_url = os.path.join(TRAIN_SET_PATH, person_data_list[index])
		print 'img_url : ' + img_url

		if detect_face(img_url, training = True) == 'SameData':
			index += 1
		else:
			response = hanldle_error_generally(CF.person.create, PERSON_GROUP_ID, name)
			print response
			person_id = response['personId']
			try:
				CF.person.add_face(img_url, PERSON_GROUP_ID, person_id)
				batch_data_list.append(person_data_list[index])
				person_num += 1
				index += 1
				print 'person : ' + str(person_num)
			except CF.CognitiveFaceException as e:
				if e.code == 'RateLimitExceeded':
					time.sleep(1)
				elif e.code == 'BadArgument':
					print e.code
					index += 1


def train(people_data_list, person_group_id_list, add_face_switch=True):
	if PERSON_GROUP_ID not in person_group_id_list or add_face_switch == False:
		create_person_group(person_group_id_list)
	batch_data_list, person_data_list = list(), list()
	name = people_data_list[0].split('_')[0]
	num = 1
	for person_data in people_data_list:
		if num == len(people_data_list):
			person_data_list.append(person_data)
			add_face_for_batch_size(name, person_data_list, batch_data_list)	

		elif name == person_data.split('_')[0]:
			person_data_list.append(person_data)
		else:
			add_face_for_batch_size(name, person_data_list, batch_data_list)

			name = person_data.split('_')[0]
			person_data_list = list()
			person_data_list.append(person_data)
		num += 1
	added_face_list = hanldle_error_generally(CF.person.lists,PERSON_GROUP_ID)
	print 'added_face_list num : ' + str(len(added_face_list))

	print batch_data_list

	hanldle_error_generally(CF.person_group.train, PERSON_GROUP_ID)
	response = hanldle_error_generally(CF.person_group.get_status, PERSON_GROUP_ID)
	status = response['status']

	print status

def practice(test_people_data_list):
	error_num = 0
	step = 0
	index = 0
	correct = 0
	while step != STEPS and index != len(test_people_data_list):
		img = test_people_data_list[index]
		test_img_url = os.path.join(TEST_SET_PATH, img)
		try:
			detection_faces = detect_face(test_img_url)
			print '______________________'
			print '<step ' + str(step) + '>    img name : ' + img
			face_num = 0
			if detection_faces != 'SameData':
				for detection_face in detection_faces:
					label = img.split('_')[0]
					if detection_face == label:
						face_num += 1
						print str(face_num)+ '. face '
						print('detection_face : '+detection_face+'\tlabel : '+label)
						correct += 1
					elif detection_face == None:
						face_num += 1
						print str(face_num)+ '. face '
						print 'new_person'
						##TODO: make add_new_person_function
					else:
						print('detection_face : '+detection_face+'\tlabel : '+label)
						print 'wrong'
					print 
			step += 1
			index += 1
		except CF.CognitiveFaceException as e:
			print(e.code)
			if e.code == 'BadArgument':
				index += 1
				error_num += 1
			elif e.code == 'RateLimitExceeded':
				time.sleep(5)
	print '---------------------'
	print "correct : " + str(correct)
	acc = float(correct)/float(step)
	print "acc : {:4f}".format(acc)
	print "detection error num : " + str(error_num)




def main():
	#train_data
	people_data_list = os.listdir(TRAIN_SET_PATH)
	if '.DS_Store' in people_data_list:
		people_data_list.remove('.DS_Store')
	people_data_list.sort()

	#test_data
	test_people_data_list = os.listdir(TEST_SET_PATH)
	if '.DS_Store' in test_people_data_list:
		test_people_data_list.remove('.DS_Store')
	shuffle(test_people_data_list)

	person_group_list = hanldle_error_generally(CF.person_group.lists)
	person_group_id_list = get_person_group_id_list(person_group_list)
	print person_group_id_list

	# train(people_data_list, person_group_id_list, add_face_switch=True)
	train(people_data_list, person_group_id_list, add_face_switch=False)

	practice(test_people_data_list)


main()
# faces = CF.face.detect(test_img_url)

# def getRectangle(faceDictionary):
#     rect = faceDictionary['faceRectangle']
#     left = rect['left']
#     top = rect['top']
#     bottom = left + rect['height']
#     right = top + rect['width']
#     return ((left, top), (bottom, right))

# img = Image.open(test_img_url)
# print(img)
# draw = ImageDraw.Draw(img)
# for face in faces:
#     draw.rectangle(getRectangle(face), outline='red')
# img.show()

