#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import sys
import warnings
from timeit import time

import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

import settings
from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection as ddet
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

#from darknet import darknet

warnings.filterwarnings('ignore')


def main(yolo):

   # Definition of the parameters
    max_cosine_distance = settings.MAX_COSINE_DISTANCE
    nn_budget = settings.NN_BUDGET
    nms_max_overlap = settings.NMS_MAX_OVERLAP
    
   # deep_sort 
    encoder = gdet.create_box_encoder(settings.MODEL_FILENAME, batch_size=4)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine', settings.MAX_COSINE_DISTANCE, settings.NN_BUDGET)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture(settings.VIDEO_IN) # feed testing video
    #video_capture = cv2.VideoCapture('round.mp4')
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    #print ("Frames per second".format(video_fps))

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*settings.CODEC)
        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#cv2.VideoWriter
        out = cv2.VideoWriter(settings.VIDEO_OUT, fourcc, video_fps, (w, h))
        #out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open(settings.TRACKING_FILE, 'w')
        frame_index = -1 
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1]) # bgr to rgb
        boxs = yolo.detect_image(image)
        #boxs = darknet.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature)
                      for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, settings.NMS_MAX_OVERLAP, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        bbox_center = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 

            bbox = track.to_tlbr()  # generate_detections.py, input is a BGR color image
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[2]), int(bbox[3])),
                          (0, 0, 255), 2)  # red color 
            cv2.putText(frame, str(track.track_id),
                        (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200,
                        (0, 255, 0), 2)  # RGB, Green color 
            bbox_center = (str(track.track_id), [int(bbox[0] + bbox[2]) / 2,
                                                 int(bbox[1] + bbox[3]) / 2])
            print(bbox_center)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            for track in tracker.tracks:
                bbox = track.to_tlbr()                                                 
                bbox_center_o = ([int(bbox[0] + bbox[2]) / 2,
                                  int(bbox[1] + bbox[3]) / 2])
                list_file.write(str(track.track_id) + ','
                                + str(bbox_center_o) + '; ')
            
            list_file.write('\n')
            
            
        """
        # don't need detections
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]),
                int(bbox[3])),(255,0,0), 2)  # RGB, (0, 0, 255), blue color
            
        cv2.imshow('', frame)
        """

        fps  = (fps + (1. / (time.time() - t1))) / 2
        print(f'fps = {fps}')
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main(YOLO())
