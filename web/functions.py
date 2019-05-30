import models
import sqlite3
import numpy as np
import cv2
import os 
import glob
from time import sleep 
import random 
import tensorflow as tf 
import facenet
import align.detect_face
from scipy import misc
from models import *

class connect_db:
  def __enter__(self):
    self.conn = sqlite3.connect('users.sqlite')
    self.c = self.conn.cursor()
    return self.c

  def __exit__(self, *exc):
    self.conn.commit()
    self.c.close()
    self.conn.close()

def get_users():
  with connect_db() as cursor:
    users = []
    sql = '''
      SELECT
        A.user_id,
        A.name,
        A.email,
        B.type,
        B.date
      FROM
        user A
      NATURAL JOIN
        (
          SELECT
            MAX(attendance_id) AS id,
            date,
            type,
            user_id AS uid
          FROM
            attendance
          GROUP BY 
            uid
        ) B
      WHERE 
        A.user_id = B.uid 
    '''
    for row in cursor.execute(sql):
      row = list(row)
      row[3] = 'Working' if row[3] == 'I' else 'Leaved'
      users.append(models.User(*row))
    return users

def get_user(user_id):
  with connect_db() as cursor:
    sql = '''
      SELECT
        A.user_id,
        A.name,
        A.email,
        B.type,
        B.date
      FROM
        user A
      NATURAL JOIN
        (
          SELECT
            MAX(attendance_id) AS id,
            date,
            type,
            user_id AS uid
          FROM
            attendance
          GROUP BY 
            uid
        ) B
      WHERE 
        A.user_id = B.uid 
      AND 
        A.user_id = ?
    '''
    cursor.execute(sql, (user_id,))
    row = list(cursor.fetchone())
    row[3] = 'Working' if row[3] == 'I' else 'Leaved'
    return models.User(*row)

def create_user(name, email):
  with connect_db() as cursor:
    sql = 'INSERT INTO user (name, email) VALUES (?, ?)'
    cursor.execute(sql, (name, email, ))
    sql = '''
      SELECT 
        user_id 
      FROM 
        user
      ORDER BY 
        user_id 
      DESC 
      LIMIT 1
    '''
    cursor.execute(sql)
    user_id = cursor.fetchone()[0]
    sql = '''
      INSERT INTO 
        attendance (
          user_id, 
          date, 
          type
        ) 
      VALUES (
        ?, 
        DATETIME(),
        'O'
      )
      '''
    cursor.execute(sql)
    return user_id

def update_user(user_id, body):
  with connect_db() as cursor:
    if 'email' not in body.keys() or 'name' not in body.keys():
      raise ValueError('Incorrect POST body - some values are missing')
    sql = 'SELECT * FROM user WHERE user_id = ?'
    cursor.execute(sql, (user_id, ))
    if cursor.fetchone() == None:
      raise ValueError('Invalid ID value')
    sql = 'UPDATE user SET user = ?, email = ? WHERE user_id = ?'
    cursor.execute(sql, (body['user'], body['email'], user_id, ))

def get_attendances():
  with connect_db() as cursor:
    attendances = []
    sql = '''
      SELECT
        A.attendance_id,
        B.user_id,
        A.date,
        A.type,
        B.name
      FROM
        attendance A
      NATURAL JOIN
        user B
      ORDER BY
        attendance_id DESC
      LIMIT
        100
    '''
    for row in cursor.execute(sql):
      row = list(row)
      row[3] = 'Entered' if row[3] == 'I' else 'Leaved'
      attendances.append(models.Attendance(*row))
    return attendances

def get_attendance(att_id):
  with connect_db() as cursor:
    sql = '''
      SELECT
        attendance_id,
        user_id,
        date.
        type
      FROM
        attendance
      WHERE
        attendance_id = ?
    '''
    cursor.execute(sql, (att_id, ))
    row = cursor.fetchone()
    return models.Attendance(*row)

def update_attendance(user_id):
  user = get_user(user_id)
  update_type = 'O' if user.status == 'Working' else 'I'

  with connect_db() as cursor:
    sql = '''
      INSERT INTO
        attendance (
          user_id,
          date,
          type
      ) VALUES (
        ?,
        DATETIME(),
        ?
      )
    '''
    cursor.execute(sql, (user_id, update_type, ))
    

def align_dataset_mtcnn(target, croptype, image_size = 160, margin = 44, random_order = 'store_true', gpu_memory_fraction = 1.0, detect_multiple_faces = True, text_counter = 0):
  with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
      pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
  minsize = 20 # minimum size of face
  threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
  factor = 0.709 # scale factor

  bounding_box_list = []
    
  output_class_dir = os.path.join('raw_' + croptype, target)
  files = list(filter(lambda x: 'DS_Store' not in x, sorted(os.listdir(output_class_dir))))
  if len(files) == 0:
    print('fatal: no image in target folder')
  filename = files[0]
  filepath = os.path.join(output_class_dir, filename)
  try:
    img = misc.imread(filepath)
  except (IOError, ValueError, IndexError) as e:
    errorMessage = '{}: {}'.format(filepath, e)
    print('fatal:', errorMessage)
    return 

  if img.ndim < 2:
    print('fatal:', 'Unable to align "%s"' % filepath)
    return
  if img.ndim == 2:
    img = facenet.to_rgb(img)
  img = img[:,:,0:3]

  bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
  nrof_faces = bounding_boxes.shape[0]
  if nrof_faces > 0:
    det = bounding_boxes[:,0:4]
    det_arr = []
    img_size = np.asarray(img.shape)[0:2]
    if nrof_faces>1:
      if detect_multiple_faces:
        for i in range(nrof_faces):
          det_arr.append(np.squeeze(det[i]))
      else:
        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
        img_center = img_size / 2
        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
        det_arr.append(det[index,:])
    else:
      det_arr.append(np.squeeze(det))

    for i, det in enumerate(det_arr):
      det = np.squeeze(det)
      bb = np.zeros(4, dtype=np.int32)
      bb[0] = np.maximum(det[0]-margin/2, 0)
      bb[1] = np.maximum(det[1]-margin/2, 0)
      bb[2] = np.minimum(det[2]+margin/2, img_size[1])
      bb[3] = np.minimum(det[3]+margin/2, img_size[0])
      cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
      scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')

      bounding_box_list.append(bb[:4])
  else:
    print('fatal:', 'Unable to align "%s"' % filepath)

  return filename, bounding_box_list

def tracking(target, croptype, bounding_boxes, input_dir):
  cap = cv2.imread('raw_{}/{}/{}'.format(croptype, input_dir, target))
  track_windows = []

  for box in bounding_boxes:
    print(box)
    x1, y1, x2, y2 = box
    r, h, c, w = y1, y2 - y1, x1, x2 - x1
    track_window = (c, r, w, h)
    track_windows.append(track_window)

  listOfPic = list(sorted(filter(lambda x: '.jpg' in x, os.listdir(os.path.join('raw_' + croptype, input_dir)))))
  face_dir_number = 0

  for tw, face_dir_number in zip(track_windows, range(1, len(track_windows) + 1)):
    roi = cap[tw[1] : tw[1] + tw[3], tw[0] : tw[0] + tw[2]]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


    temp_tw = tw
    if not os.path.isdir(os.path.join(croptype, input_dir)):
      os.mkdir(os.path.join(croptype, input_dir))

    print(listOfPic)
    success = 0
    failed = 0
    for pic, face_file_number in zip(listOfPic, range(1, len(listOfPic) + 1)):
      frame = cv2.imread('raw_{}/{}/{}'.format(croptype, input_dir, pic))
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

      ret, temp_tw = cv2.CamShift(dst, temp_tw, term_crit)
      ret1 = (ret[0], ret[1], 0)

      pts = cv2.boxPoints(ret1)
      pts = np.int0(pts)

      cropped = frame[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]
      if cropped.size == 0:
        print('Something\'s wrong with file', 'input_dir/{}'.format(pic) )
        failed += 1
        continue
      resize_crop = cv2.resize(cropped, dsize=(160, 160), interpolation=cv2.INTER_AREA)

      cv2.imwrite((croptype +  '/' + input_dir + '/face' + str(face_dir_number) + '_' + str(face_file_number) + '.jpg'), resize_crop)
      success += 1
    print('success: {}, failed: {}'.format(success, failed))