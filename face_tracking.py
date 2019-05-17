import numpy as np
import cv2
import os
import glob
     

def tracking(bounding_boxes_filename, saving_path, input_dir):
    with open(bounding_boxes_filename, "r") as text_file:
        while True:
            line = text_file.readline()
            if not line:
                break
            elif line[0] == 'C':
                track_windows = list()
                line = line.replace("\\",'/')
                line = line.replace("\t",'/t')
                line = line.replace("\n",'')
                cap = cv2.imread(line)
            else:
                x1, y1, x2, y2 = line.split(' ')
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                r, h, c, w = y1, y2 - y1, x1, x2 - x1
                track_window = (c, r, w, h)
                track_windows.append(track_window)


    listOfPic = glob.glob(input_dir + "/Day3/*.jpg")
    face_dir_number = 0

    for tw in track_windows:
        roi = cap[tw[1] : tw[1] + tw[3], tw[0] : tw[0] + tw[2]]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


        face_file_number = 0
        temp_tw = tw
        if not os.path.exists(saving_path + 'face' + str(face_dir_number)):
            os.makedirs(saving_path + 'face' + str(face_dir_number))

        for pic in listOfPic:
            frame = cv2.imread(pic)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            ret, temp_tw = cv2.CamShift(dst, temp_tw, term_crit)
            ret1 = (ret[0], ret[1], 0)

            pts = cv2.boxPoints(ret1)
            pts = np.int0(pts)

            cropped = frame[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]
            resize_crop = cv2.resize(cropped, dsize=(160, 160), interpolation=cv2.INTER_AREA)

            cv2.imwrite((saving_path + 'face' + str(face_dir_number) + '/face' + str(face_file_number) + '.jpg'), resize_crop)
            face_file_number += 1

        face_dir_number += 1
