from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np

import align.detect_face
from for_detection import *
import random
from time import sleep, clock

def align_dataset_mtcnn(input_dir, output_dir, image_size = 160, margin = 44, random_order = 'store_true', gpu_memory_fraction = 1.0, detect_multiple_faces = True, text_counter = 0):
    sleep(random.random())
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset = get_dataset(input_dir)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    bounding_boxes_dir = os.path.join(output_dir, 'bounding_boxes_txt')
    if not os.path.exists(bounding_boxes_dir):
        os.makedirs(bounding_boxes_dir)
    bounding_boxes_filename = os.path.join(bounding_boxes_dir, 'bounding_boxes_%d.txt' % text_counter)
    
    with open(bounding_boxes_filename, "w") as text_file:
        
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        for cls in dataset:
            #이거 나중에 지우기
            text_file.write('%snext_line' % cls.image_paths[0])
            output_class_dir = os.path.join(output_dir, cls.name)
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(cls.image_paths[0])[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            print(cls.image_paths[0])
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(cls.image_paths[0])
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(cls.image_paths[0], e)
                    print(errorMessage)
                else:
                    if img.ndim<2:
                        print('Unable to align "%s"' % cls.image_paths[0])
                        text_file.write('%snext_line' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:,:,0:3]

                    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces>0:
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
                            nrof_successfully_aligned += 1
                            filename_base, file_extension = os.path.splitext(output_filename)
                            '''
                            if detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
                            misc.imsave(output_filename_n, scaled)
                            '''
                            text_file.write('%d %d %d %dnext_line' % (bb[0], bb[1], bb[2], bb[3]))
                    else:
                        print('Unable to align "%s"' % cls.image_paths[0])
                        text_file.write('%s\n' % (output_filename))
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)