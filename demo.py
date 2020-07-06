#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

#from darknet import darknet

warnings.filterwarnings('ignore')

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.25
    nn_budget = None
    nms_max_overlap = 0.1
    
   # deep_sort 
    model_filename = 'model_data/veri.pb'

    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture('test_scene5.mp4') # feed testing video
    #video_capture = cv2.VideoCapture('round.mp4')
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    #print ("Frames per second".format(video_fps))

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#cv2.VideoWriter
        out = cv2.VideoWriter('test_scene5_tar.mp4', fourcc, video_fps, (w, h))
        #out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('tracking_DJI_0006.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
        #boxs = darknet.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        bbox_center = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()                                                  # generate_detections.py, input is a BGR color image
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0, 0, 255), 2)  # red color 
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)    # RGB, Green color 
            bbox_center = (str(track.track_id), [int(bbox[0] + bbox[2])/2, int(bbox[1] + bbox[3])/2])
            print(bbox_center)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            
            for track in tracker.tracks:
                bbox = track.to_tlbr()                                                 
                bbox_center_o = ([int(bbox[0] + bbox[2])/2, int(bbox[1] + bbox[3])/2])
                
                list_file.write(str(track.track_id) + ',' + str(bbox_center_o) + '; ')
            
            list_file.write('\n')
            
            
        """
        # don't need detections
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)  # RGB, (0, 0, 255), blue color
            
        cv2.imshow('', frame)
        """

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
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

