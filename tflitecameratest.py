import numpy as np
import cv2
import os
import cv2
import tensorflow as tf

# Load the TensorFlow Lite models
pnet_model_path = './tflite/pnet_model.tflite'
rnet_model_path = './tflite/rnet_model.tflite'
onet_model_path = './tflite/onet_model.tflite'

# Load the TensorFlow Lite interpreter for each model
pnet_interpreter = tf.lite.Interpreter(model_path=pnet_model_path)
pnet_interpreter.allocate_tensors()

rnet_interpreter = tf.lite.Interpreter(model_path=rnet_model_path)
rnet_interpreter.allocate_tensors()

onet_interpreter = tf.lite.Interpreter(model_path=onet_model_path)
onet_interpreter.allocate_tensors()

def preprocess_image(image, target_shape):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_shape[:2])  # Resize the image
    image = (np.float32(image) - 127.5) / 127.5  # Normalize the image
    return np.expand_dims(image, axis=0)

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Preprocess the frame for each model
    pnet_input = preprocess_image(frame, (12, 12, 3))  # Adjust target width/height for P-Net
    rnet_input = preprocess_image(frame, (24, 24, 3))  # Fixed size for R-Net
    onet_input = preprocess_image(frame, (48, 48, 3))  # Fixed size for O-Net


    # Set the input tensors for each interpreter
    pnet_interpreter.set_tensor(pnet_interpreter.get_input_details()[0]['index'], pnet_input)
    rnet_interpreter.set_tensor(rnet_interpreter.get_input_details()[0]['index'], rnet_input)
    onet_interpreter.set_tensor(onet_interpreter.get_input_details()[0]['index'], onet_input)

    # Run inference for each model
    pnet_interpreter.invoke()
    rnet_interpreter.invoke()
    onet_interpreter.invoke()

    # Get the output tensors for each interpreter
    pnet_output = pnet_interpreter.get_tensor(pnet_interpreter.get_output_details()[0]['index'])
    rnet_output = rnet_interpreter.get_tensor(rnet_interpreter.get_output_details()[0]['index'])
    onet_output = onet_interpreter.get_tensor(onet_interpreter.get_output_details()[0]['index'])

    # Process the output tensors for P-Net
    pnet_boxes = pnet_output[0]  # Extract bounding boxes
    pnet_scores = pnet_output[1]  # Extract confidence scores

    # Filter out low-confidence detections
    pnet_threshold = 0.5  # You can adjust this threshold
    pnet_indices = np.where(pnet_scores > pnet_threshold)[0]

    # Extract boxes and scores based on filtered indices
    pnet_filtered_boxes = pnet_boxes[pnet_indices]
    pnet_filtered_scores = pnet_scores[pnet_indices]

    # Process the output tensors for R-Net
    rnet_boxes = rnet_output[0]  # Extract bounding boxes
    rnet_scores = rnet_output[1]  # Extract confidence scores

    # Filter out low-confidence detections
    rnet_threshold = 0.5  # You can adjust this threshold
    rnet_indices = np.where(rnet_scores > rnet_threshold)[0]

    # Extract boxes and scores based on filtered indices
    rnet_filtered_boxes = rnet_boxes[rnet_indices]
    rnet_filtered_scores = rnet_scores[rnet_indices]

    # Process the output tensors for O-Net
    onet_boxes = onet_output[0]  # Extract bounding boxes
    onet_scores = onet_output[1]  # Extract confidence scores
    onet_landmarks = onet_output[2]  # Extract landmark points

    # Filter out low-confidence detections
    onet_threshold = 0.5  # You can adjust this threshold
    onet_indices = np.where(onet_scores > onet_threshold)[0]

    # Extract boxes, scores, and landmarks based on filtered indices
    onet_filtered_boxes = onet_boxes[onet_indices]
    onet_filtered_scores = onet_scores[onet_indices]
    onet_filtered_landmarks = onet_landmarks[onet_indices]

    # Draw bounding boxes and landmarks for P-Net
    for i in range(pnet_filtered_boxes.shape[0]):
        bbox = pnet_filtered_boxes[i]
        score = pnet_filtered_scores[i]

        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw bounding boxes for R-Net
    for i in range(rnet_filtered_boxes.shape[0]):
        bbox = rnet_filtered_boxes[i]
        score = rnet_filtered_scores[i]

        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (0, 255, 0), 1)
        cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw bounding boxes and landmarks for O-Net
    for i in range(onet_filtered_boxes.shape[0]):
        bbox = onet_filtered_boxes[i]
        score = onet_filtered_scores[i]

        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (0, 0, 255), 1)
        cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw landmarks for O-Net if available
        if onet_filtered_landmarks.shape[0] > 0:
            landmarks = onet_filtered_landmarks[i]
            for j in range(len(landmarks) // 2):
                x = int(landmarks[j * 2])
                y = int(landmarks[j * 2 + 1])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # -1 to fill the circle

    # Display the processed frame
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
