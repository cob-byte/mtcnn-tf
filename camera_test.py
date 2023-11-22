#coding:utf-8
import sys
import argparse
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from detection.MtcnnDetector import MtcnnDetector
import cv2
import os
import numpy as np 

test_mode = "ONet"
thresh = [0.1, 0.15, 0.6]
min_face_size = 20
stride = 2
detectors = [None, None, None]
prefix = ['./tmp/model/pnet/pnet', './tmp/model/rnet/rnet', './tmp/model/onet/onet']
epoch = [30, 30, 2]
batch_size = [2048, 64, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]


PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh)

# Open a connection to the camera
video_capture = cv2.VideoCapture(0)  # 0 for default camera, change if multiple cameras
corpbbox = None
# Set the resolution of the camera
while True:
    t1 = cv2.getTickCount()
    ret, frame = video_capture.read()  # Capture frame-by-frame

    if not ret:
        print("Failed to capture frame")
        break

    # Perform face detection on the frame
    boxes_c, landmarks = mtcnn_detector.detect(frame)
    t2 = cv2.getTickCount()
    t = (t2 - t1) / cv2.getTickFrequency()
    fps = 1.0 / t

    # Draw bounding boxes and landmarks on the frame
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)

        for j in range(len(landmarks[i]) // 2):
            cv2.circle(frame, (int(landmarks[i][2 * j]), int(landmarks[i][2 * j + 1])), 2, (0, 0, 255))

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
video_capture.release()
cv2.destroyAllWindows()