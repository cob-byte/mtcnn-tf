#coding:utf-8
import math
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
import mediapipe as mp

test_mode = "ONet"
thresh = [0.1, 0.15, 0.7]
min_face_size = 20
stride = 2
detectors = [None, None, None]
prefix = ['./tmp/model/pnet/pnet', './tmp/model/rnet/rnet', './tmp/model/onet/onet']
epoch = [30, 30, 30]
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

mpFace = mp.solutions.face_mesh 
face = mpFace.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3, min_tracking_confidence=0.3)
multi_face_landmarks = []

# Open a connection to the camera
video_capture = cv2.VideoCapture(0)  # 0 for default camera, change if multiple cameras
corpbbox = None

def get_top_lip_height(landmarks):
    # 39 -> 81
    # 0 -> 13
    # 269 -> 311

    p39 = np.array([landmarks[39].x, landmarks[39].y, landmarks[39].z])
    p81 = np.array([landmarks[81].x, landmarks[81].y, landmarks[81].z])
    p0 = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    p13 = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
    p269 = np.array([landmarks[269].x, landmarks[269].y, landmarks[269].z])
    p311 = np.array([landmarks[311].x, landmarks[311].y, landmarks[311].z])

    d1 = np.linalg.norm(p39-p81)
    d2 = np.linalg.norm(p0-p13)
    d3 = np.linalg.norm(p269-p311)

    # print("average: ", (d1 + d2 + d3) / 3)
    return  (d1 + d2 + d3) / 3

    
def get_bottom_lip_height(landmarks):
    # 181 -> 178
    # 17 -> 14
    # 405 -> 402

    p181 = np.array([landmarks[181].x, landmarks[181].y, landmarks[181].z])
    p178 = np.array([landmarks[178].x, landmarks[178].y, landmarks[178].z])
    p17 = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    p14 = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    p405 = np.array([landmarks[405].x, landmarks[405].y, landmarks[405].z])
    p402 = np.array([landmarks[402].x, landmarks[402].y, landmarks[402].z])

    d1 = np.linalg.norm(p181-p178)
    d2 = np.linalg.norm(p17-p14)
    d3 = np.linalg.norm(p405-p402)

    # print("average: ", (d1 + d2 + d3) / 3)
    return  (d1 + d2 + d3) / 3


def get_mouth_height(landmarks):
    # 178 -> 81
    # 14 -> 13
    # 402 -> 311

    p178 = np.array([landmarks[178].x, landmarks[178].y, landmarks[178].z])
    p81 = np.array([landmarks[81].x, landmarks[81].y, landmarks[81].z])
    p14 = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    p13 = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
    p402 = np.array([landmarks[402].x, landmarks[402].y, landmarks[402].z])
    p311 = np.array([landmarks[311].x, landmarks[311].y, landmarks[311].z])

    d1 = np.linalg.norm(p178-p81)
    d2 = np.linalg.norm(p14-p13)
    d3 = np.linalg.norm(p402-p311)

    # print("average: ", (d1 + d2 + d3) / 3)
    return  (d1 + d2 + d3) / 3

def check_mouth_open(landmarks):
    top_lip_height =    get_top_lip_height(landmarks)
    bottom_lip_height = get_bottom_lip_height(landmarks)
    mouth_height =      get_mouth_height(landmarks)

    # if mouth is open more than lip height * ratio, return true.
    ratio = 0.8
    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return 1
    else:
        return 0

def euclidean(point, point1):
    x = point.x
    y = point.y
    x1 = point1.x
    y1 = point1.y

    return math.sqrt((x1 - x)**2 + (y1 - y)**2)

def closed_ratio(landmarks, left_eye_indices, right_eye_indices):
    rh_right = landmarks[right_eye_indices[0]]
    rh_left = landmarks[right_eye_indices[8]]
    rv_top = landmarks[right_eye_indices[12]]
    rv_bottom = landmarks[right_eye_indices[4]]

    lh_right = landmarks[left_eye_indices[0]]
    lh_left = landmarks[left_eye_indices[8]]
    lv_top = landmarks[left_eye_indices[12]]
    lv_bottom = landmarks[left_eye_indices[4]]

    rhDistance = euclidean(rh_right, rh_left)
    rvDistance = euclidean(rv_top, rv_bottom)
    lvDistance = euclidean(lv_top, lv_bottom)
    lhDistance = euclidean(lh_right, lh_left)
    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance
    ratio = (reRatio + leRatio) / 2

    return ratio

def check_eyes_open(landmarks, left_eye_indices, right_eye_indices):
    eyes_closed_ratio = closed_ratio(landmarks, left_eye_indices, right_eye_indices)
    ratio_threshold = 5
    if eyes_closed_ratio > ratio_threshold:
        return 0  # closed
    else:
        return 1  # open
    
# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

def get_orientation(landmarks):
    image_points = np.array([
        (landmarks[0][2*2], landmarks[0][2*2+1]),     # Nose tip
        ((landmarks[0][3*2] + landmarks[0][4*2]) / 2, (landmarks[0][3*2+1] + landmarks[0][4*2+1]) / 2),  # Chin
        (landmarks[0][0*2], landmarks[0][0*2+1]),     # Left eye left corner
        (landmarks[0][1*2], landmarks[0][1*2+1]),     # Right eye right corner
        (landmarks[0][3*2], landmarks[0][3*2+1]),     # Left Mouth corner
        (landmarks[0][4*2], landmarks[0][4*2+1])      # Right mouth corner
    ], dtype="double")

    # Solve for pose
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # Project a 3D point (0, 0, 1000.0) onto the image plane to calculate pitch, yaw and roll angles
    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    print(angle)
    # Determine the orientation based on the angle
    if -40 < angle < 0:
        return "Looking left"
    elif -200 < angle < -140:
        return "Looking right"
    elif -139.99 < angle < -102:
        return "Looking up"
    elif -101 < angle < -60:
        return "Looking down"
    else:
        return "Looking straight"

# Set the resolution of the camera
while True:
    t1 = cv2.getTickCount()
    ret, frame = video_capture.read()  # Capture frame-by-frame

    if not ret:
        print("Failed to capture frame")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection on the frame
    boxes_c, landmarks = mtcnn_detector.detect(image_rgb)

    if len(landmarks) != 0:
        # Camera internals
        focal_length = frame.shape[1]
        center = (frame.shape[1]/2, frame.shape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Calculate the orientation
        orientation = get_orientation(landmarks)
        cv2.putText(frame, orientation, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    t2 = cv2.getTickCount()
    t = (t2 - t1) / cv2.getTickFrequency()
    fps = 1.0 / t

    # Draw bounding boxes and landmarks on the frame
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

        for j in range(len(landmarks[i]) // 2):
            cv2.circle(frame, (int(landmarks[i][2 * j]), int(landmarks[i][2 * j + 1])), 2, (0, 0, 255))
        
        cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)

    # Face Mesh
    results = face.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [lm for lm in face_landmarks.landmark]

            # Check if eyes are open or closed
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            eyes_are_open = check_eyes_open(landmarks, left_eye_indices, right_eye_indices)

            # Check if mouth is open
            mouth_is_open = check_mouth_open(landmarks)
            
            # Display text based on eye and mouth states
            eye_status_text = "Eyes: Closed" if eyes_are_open == 0 else "Eyes: Open"
            mouth_status_text = "Mouth: Closed" if mouth_is_open == 0 else "Mouth: Open"
            cv2.putText(frame, eye_status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color
            cv2.putText(frame, mouth_status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
face.close()
video_capture.release()
cv2.destroyAllWindows()