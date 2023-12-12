import cv2
from collections import deque
import numpy as np
import statistics
import time
import logging
import math
import mediapipe as mp
import pyrebase
import datetime
from tools.loader import TestLoader
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from detection.MtcnnDetector import MtcnnDetector
from training.mtcnn_model import P_Net, R_Net, O_Net

config = {
    "apiKey": "none",
    "authDomain": "none",
    "databaseURL": "none",
    "storageBucket": "none",
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()

class FaceDetection:
    def __init__(self):
        self.detectors = [None, None, None]
        test_mode = "ONet"
        thresh = [0.1, 0.15, 0.5]
        min_face_size = 24
        stride = 2
        prefix = ['./tmp/model/pnet/pnet', './tmp/model/rnet/rnet', './tmp/model/onet/onet']
        epoch = [30, 30, 30]
        batch_size = [2048, 64, 16]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

        PNet = FcnDetector(P_Net, model_path[0])
        self.detectors[0] = PNet

        if test_mode in ["RNet", "ONet"]:
            RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
            self.detectors[1] = RNet

        if test_mode == "ONet":
            ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
            self.detectors[2] = ONet

        self.mtcnn_detector = MtcnnDetector(detectors=self.detectors, min_face_size=min_face_size,
                                            stride=stride, threshold=thresh)

    def detect_face(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes_c, landmarks = self.mtcnn_detector.detect(image_rgb)
        return boxes_c, landmarks
    
    def get_orientation(self, landmarks, camera_matrix, dist_coeffs):
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

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
        # Determine the orientation based on the angle
        if -40 < angle < 0:
            return "Left"
        elif -200 < angle < -120:
            return "Right"
        else:
            return "Straight"

class FaceCheck:
    def __init__(self, baby_monitor):
        self.baby_monitor = baby_monitor
        self.mpFace = mp.solutions.face_mesh
        self.face = self.mpFace.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.multi_face_landmarks = []
        self.eyes_open_q = deque(maxlen=30)
    
    # Decorator ensures function that can only be called once every `s` seconds.
    def debounce(s):
        def decorate(f):
            t = None

            def wrapped(*args, **kwargs):
                nonlocal t
                t_ = time.time()
                if t is None or t_ - t >= s:
                    result = f(*args, **kwargs)
                    t = time.time()
                    return result
            return wrapped
        return decorate

    def get_top_lip_height(self, landmarks):
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

    
    def get_bottom_lip_height(self, landmarks):
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


    def get_mouth_height(self, landmarks):
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

    def check_mouth_open(self, landmarks):
        top_lip_height = self.get_top_lip_height(landmarks)
        bottom_lip_height = self.get_bottom_lip_height(landmarks)
        mouth_height = self.get_mouth_height(landmarks)

        ratio = 0.8
        if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
            return 1
        else:
            return 0

    def euclidean(self, point, point1):
        x = point.x
        y = point.y
        x1 = point1.x
        y1 = point1.y

        return math.sqrt((x1 - x)**2 + (y1 - y)**2)

    def closed_ratio(self, landmarks, left_eye_indices, right_eye_indices):
        rh_right = landmarks[right_eye_indices[0]]
        rh_left = landmarks[right_eye_indices[8]]
        rv_top = landmarks[right_eye_indices[12]]
        rv_bottom = landmarks[right_eye_indices[4]]

        lh_right = landmarks[left_eye_indices[0]]
        lh_left = landmarks[left_eye_indices[8]]
        lv_top = landmarks[left_eye_indices[12]]
        lv_bottom = landmarks[left_eye_indices[4]]

        rhDistance = self.euclidean(rh_right, rh_left)
        rvDistance = self.euclidean(rv_top, rv_bottom)
        lvDistance = self.euclidean(lv_top, lv_bottom)
        lhDistance = self.euclidean(lh_right, lh_left)
        reRatio = rhDistance / rvDistance
        leRatio = lhDistance / lvDistance
        ratio = (reRatio + leRatio) / 2

        return ratio

    def check_eyes_open(self, landmarks, left_eye_indices, right_eye_indices):
        eyes_closed_ratio = self.closed_ratio(landmarks, left_eye_indices, right_eye_indices)
        ratio_threshold = 5
        if eyes_closed_ratio > ratio_threshold:
            return 0  # closed
        else:
            return 1  # open
    
    @debounce(1)
    def awake_voting_logic(self):
        if len(self.eyes_open_q) > len(self.eyes_open_q)/2: # dont vote on eyes unless queue is half full
            avg = sum(self.eyes_open_q) / len(self.eyes_open_q)
            if avg > 0.75: # eyes open
                self.eyes_open_state = True
                logging.info("\nvote awake")
                self.baby_monitor.awake_q.append(3)
            else: # closed
                self.eyes_open_state = False
                self.baby_monitor.awake_q.append(0)
                logging.info("\nvote sleeping")
        else:
            print("Not voting on eyes, eye queue too short.")
    
    @debounce(1)
    def throttled_handle_no_eyes_found(self):
        if(len(self.eyes_open_q) > 0):
            logging.info('No face found, depreciate queue')
            print('No face found, depreciate queue')
            self.eyes_open_q.popleft()
    
    def detect_facemesh(self, image_rgb, frame):
        results_face = self.face.process(image_rgb)
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                landmarks = [lm for lm in face_landmarks.landmark]

                # Check if eyes are open or closed
                LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] 

                eyes_are_open = self.check_eyes_open(landmarks, LEFT_EYE, RIGHT_EYE)

                if eyes_are_open == 0: # if eyes are closed, then check if mouth is open
                    mouth_is_open = self.check_mouth_open(landmarks)
                    if mouth_is_open:
                        logging.info('Eyes closed, mouth open, crying or yawning, consider awake.')
                        self.eyes_open_q.append(1)
                    else:
                        logging.info('Eyes closed, mouth closed, consider sleeping.')
                        self.eyes_open_q.append(0)
                else:
                    mouth_is_open = self.check_mouth_open(landmarks)
                    logging.info('Eyes open, consider awake.')
                    self.eyes_open_q.append(1)

                # Display text based on eye and mouth states
                eye_status_text = "Eyes: Closed" if eyes_are_open == 0 else "Eyes: Open"
                mouth_status_text = "Mouth: Closed" if mouth_is_open == 0 else "Mouth: Open"
                cv2.putText(frame, eye_status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color
                cv2.putText(frame, mouth_status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color

                # Draw circles around the landmarks for left and right eyes
                for idx in LEFT_EYE:
                    lm = landmarks[idx]
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw a smaller circle around left eye landmarks

                for idx in RIGHT_EYE:
                    lm = landmarks[idx]
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw a smaller circle around right eye landmarks
        else: # no face results, interpret this as baby is not in crib, i.e. awake
            self.throttled_handle_no_eyes_found()
            
class MovementTracking:
    def __init__(self, baby_monitor):
        self.baby_monitor = baby_monitor
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.mpDraw = mp.solutions.drawing_utils
        self.movement_q = deque(maxlen=10) #analyze wrist movements over a specific timeframe (6 secs), a maxlen of 2 (6 seconds / 3 seconds per update)
        self.body_found = False
    
    # Decorator ensures function that can only be called once every `s` seconds.
    def debounce(s):
        def decorate(f):
            t = None

            def wrapped(*args, **kwargs):
                nonlocal t
                t_ = time.time()
                if t is None or t_ - t >= s:
                    result = f(*args, **kwargs)
                    t = time.time()
                    return result
            return wrapped
        return decorate
    
    @debounce(1)
    def throttled_handle_no_body_found(self):
        logging.info('No body found, vote awake')
        print('No body found, vote awake')
        self.baby_monitor.awake_q.append(1)
    
    @debounce(1)
    def track_movement(self, frame, left_wrist_coords, right_wrist_coords):
        if self.body_found:  # Use self.body_found here
            self.movement_q.append((left_wrist_coords, right_wrist_coords))
            if len(self.movement_q) >= 5:
                if not self.body_found:  # Use self.body_found here
                    if len(self.movement_q):
                        self.movement_q.popleft()
                elif len(self.movement_q) > 5:
                    left_wrist_list = [c[0] for c in self.movement_q]
                    left_wrist_x_list = [c[0] for c in left_wrist_list]
                    left_wrist_y_list = [c[1] for c in left_wrist_list]

                    right_wrist_list = [c[1] for c in self.movement_q]
                    right_wrist_x_list = [c[0] for c in right_wrist_list]
                    right_wrist_y_list = [c[1] for c in right_wrist_list]

                    std_left_wrist_x = statistics.pstdev(left_wrist_x_list) - 1
                    std_left_wrist_y = statistics.pstdev(left_wrist_y_list) - 1

                    std_right_wrist_x = statistics.pstdev(right_wrist_x_list) - 1
                    std_right_wrist_y = statistics.pstdev(right_wrist_y_list) - 1

                    avg_std = (((std_left_wrist_x + std_left_wrist_y) / 2) + ((std_right_wrist_x + std_right_wrist_y) / 2)) / 2

                    if int(avg_std) < 30:
                        logging.info('No movement, vote sleeping')
                        self.baby_monitor.awake_q.append(0)
                    else:
                        logging.info("Movement, vote awake")
                        self.baby_monitor.awake_q.append(1)
            
    def detect_pose(self, image_rgb, frame):
        results_pose = self.pose.process(image_rgb)
        if results_pose.pose_landmarks:
            self.body_found = True
            # 15 left-wrist, 16 right-wrist
            shape = frame.shape
            left_wrist_coords = (shape[1] * results_pose.pose_landmarks.landmark[15].x, shape[0] * results_pose.pose_landmarks.landmark[15].y)
            right_wrist_coords = (shape[1] * results_pose.pose_landmarks.landmark[16].x, shape[0] * results_pose.pose_landmarks.landmark[16].y)

            CUTOFF_THRESHOLD = 10  # head and face
            MY_CONNECTIONS = frozenset([t for t in self.mpPose.POSE_CONNECTIONS if t[0] > CUTOFF_THRESHOLD and t[1] > CUTOFF_THRESHOLD])
            for id, lm in enumerate(results_pose.pose_landmarks.landmark):
                if id <= CUTOFF_THRESHOLD:
                    lm.visibility = 0
                    continue
                h, w,c = frame.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)

            self.mpDraw.draw_landmarks(frame, results_pose.pose_landmarks, MY_CONNECTIONS, landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            
            return left_wrist_coords, right_wrist_coords
        else:
            self.body_found = False
            self.throttled_handle_no_body_found()
    
class BabyMonitor:
    def __init__(self, tracker):
        self.tracker = tracker
        self.face_detector = FaceDetection()
        self.face_check = FaceCheck(self)
        self.movement_tracker = MovementTracking(self)
        self.awake_q = deque(maxlen=15)
        self.awake_q.append(0)
        self.face = False
        self.sleep = False
        self.is_awake = False
        self.timePutToBed = None
        self.timeFellAsleep = None
        self.wakeUpTime = None

    def process_frame(self, frame, image_rgb):
        # Get the values from the database
        self.face = db.child("devices/202010377/track/faceDetection").get().val()
        self.sleep = db.child("devices/202010377/track/trackSleeping").get().val()

        if self.face and self.sleep:
            if self.tracker.frame_count % (3 * self.tracker.video_capture.get(cv2.CAP_PROP_FPS)) == 0:
                self.face_detection(frame)
            self.track_sleeping(frame, image_rgb)
        elif self.face and not self.sleep:
            if self.tracker.frame_count % (3 * self.tracker.video_capture.get(cv2.CAP_PROP_FPS)) == 0:
                self.face_detection(frame)
        elif not self.face and self.sleep:
            self.track_sleeping(frame, image_rgb)
        else:
            print("Something went wrong.")
        
    #face detection
    def face_detection(self, frame):
        # Perform face detection
        boxes_c, landmarks = self.face_detector.detect_face(frame)
        
        if len(landmarks) == 0:
            db.child("devices/202010377/track/values/facing").set("No Face Detected")
            db.child("devices/202010377/track/values/confidence").set("No Face Detected")
            
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

            orientation_text = ""
            
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
                orientation_text = self.face_detector.get_orientation(landmarks, camera_matrix, dist_coeffs)
                
                #plot the landmarks
                for j in range(len(landmarks[i]) // 2):
                    cv2.circle(frame, (int(landmarks[i][2 * j]), int(landmarks[i][2 * j + 1])), 2, (0, 0, 255))
                
                cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, orientation_text, (corpbbox[0], corpbbox[1] -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                db.child("devices/202010377/track/values/facing").set(orientation_text)
                db.child("devices/202010377/track/values/confidence").set('{:.3f}'.format(score))
            
    #Track Sleeping
    def track_sleeping(self, frame, image_rgb):
        # Face Mesh and Pose detection
        self.face_check.detect_facemesh(image_rgb, frame)

        # Movement Tracking
        wrist_coords = self.movement_tracker.detect_pose(image_rgb, frame)
        if wrist_coords is not None:
            left, right = wrist_coords
            self.movement_tracker.track_movement(frame, left, right)

        self.face_check.awake_voting_logic()

        # Call the set_wakeness_status method periodically
        self.set_wakeness_status(frame)

    # Decorator ensures function that can only be called once every `s` seconds.
    def debounce(s):
        def decorate(f):
            t = None

            def wrapped(*args, **kwargs):
                nonlocal t
                t_ = time.time()
                if t is None or t_ - t >= s:
                    result = f(*args, **kwargs)
                    t = time.time()
                    return result
            return wrapped
        return decorate

    @debounce(10)
    def set_wakeness_status(self, img):
        if len(self.awake_q) > 0:
            avg_awake = sum(self.awake_q) / len(self.awake_q)
            if avg_awake >= 0.6 and self.is_awake == False:
                self.write_wakeness_event(True)
                print("Baby is awake")
                db.child("devices/202010377/track/values/status").set("Awake")
            elif avg_awake < 0.6 and self.is_awake == True:
                self.write_wakeness_event(False)
                print("Baby is asleep")
                db.child("devices/202010377/track/values/status").set("Asleep")
                
    @debounce(180)
    def write_wakeness_event(self, wake_status):
        if wake_status: # woke up
            if self.timeFellAsleep is not None:  # Ensure baby has fallen asleep before they can wake up
                self.is_awake = True
                self.wakeUpTime = datetime.datetime.now().strftime("%H:%M")
                self.save_and_reset()
            else:
                self.is_awake = True
        else: # fell asleep
            self.timeFellAsleep = datetime.datetime.now().strftime("%H:%M")
            self.is_awake = False

    def save_and_reset(self):
        if self.timePutToBed is not None and self.timeFellAsleep is not None and self.wakeUpTime is not None:
            # Save to database
            date = self.tracker.date  # Use the date when the tracking started
            timestamp = str(int(time.time()))
            db.child(f"devices/202010377/tracker/{date}/{timestamp}").set({
                "timePutToBed": self.timePutToBed,
                "timeFellAsleep": self.timeFellAsleep,
                "wakeUpTime": self.wakeUpTime
            })
            # Reset times
            self.timeFellAsleep = None
            self.wakeUpTime = None

class BabyTracker:
    def __init__(self):
        self.baby_monitor = BabyMonitor(self)
        self.started = False
        self.video_capture = None
        self.frame_count = 0
        self.date = None

    def check_tracking_status(self):
        self.started = db.child("devices/202010377/track/isTracking").get().val()

    def main_loop(self):
        try:
            while True:
                self.check_tracking_status()

                if self.started and self.video_capture is None:
                    self.video_capture = cv2.VideoCapture(0)  # Start Camera
                    if not self.video_capture.isOpened():
                        print("Failed to open video capture")
                        return

                elif not self.started and self.video_capture is not None:
                    self.video_capture.release()  # Stop Camera
                    self.video_capture = None
                    self.baby_monitor.timePutToBed = None
                    self.baby_monitor.timeFellAsleep = None
                    self.baby_monitor.wakeUpTime = None
                    cv2.destroyAllWindows()

                if self.video_capture is not None:
                    self.date = datetime.datetime.now().strftime("%Y-%m-%d")
                    self.baby_monitor.timePutToBed = datetime.datetime.now().strftime("%H:%M")
                    
                    ret, frame = self.video_capture.read()

                    if not ret:
                        print("Failed to capture frame")
                        break

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.baby_monitor.process_frame(frame, image_rgb)
                    
                    #save every 5 seconds
                    if self.frame_count % (3 * self.video_capture.get(cv2.CAP_PROP_FPS)) == 0:
                        # Save the processed image
                        cv2.imwrite('tracked.jpg', frame)

                        #save to database yung saved frame
                        storage.child("images/tracked.jpg").put("tracked.jpg")

                        #compare the values and do some calculations if nagising yung baby or on the verge on rolling on his/her stomach na

                        #track sleep

                        #send notification if may thresholds na nareach

                    cv2.imshow('Face Detection', frame)

                    #stop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    self.frame_count += 1
                else:
                    time.sleep(1)  # Sleep for a while before checking again

        except IOError as io_err:
            print(f"Camera error: {io_err}. Please check camera availability.")
        except Exception as ex:
            print(f"Unexpected error occurred: {ex}.")
        finally:
            # Release the camera resource in case of any exception
            if self.video_capture is not None and self.video_capture.isOpened():
                self.video_capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main_app = BabyTracker()
    main_app.main_loop()
