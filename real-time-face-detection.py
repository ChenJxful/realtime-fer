import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from PIL import Image
import time

# Create a directory to save detected face images
save_dir = 'detected-faces'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Initialize ONNX emotion predictor
from onnx_inference import ONNXEmotionPredictor
emotion_predictor = ONNXEmotionPredictor("./emotion_model.onnx")

# Open the camera
cap = cv2.VideoCapture(0)

# Create face detector using context manager
with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.7) as face_detection:

    # Initialize frame counter and current emotion
    frame_counter = 0
    current_emotion = "Unknown"
    fps = 0
    prev_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Calculate time difference between current and previous frame
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        
        # Calculate FPS
        if elapsed_time > 0:
            fps = 1 / elapsed_time
        
        # Flip the image horizontally
        img = cv2.flip(img, 1)
        
        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ensure the image is a contiguous block of memory
        img_rgb = np.ascontiguousarray(img_rgb)
        
        # Perform face detection
        results = face_detection.process(img_rgb)
        
        # Create a copy of the image for display
        display_img = img.copy()
        
        # If faces are detected
        if results.detections:
            for i, detection in enumerate(results.detections):
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Draw a green rectangle on the display image
                cv2.rectangle(display_img, bbox, (0, 255, 0), 2)
                
                # Perform emotion recognition every 3 frames
                if frame_counter % 3 == 0:
                    # Extract face region from the original image
                    x, y, w, h = bbox
                    face_img = img[y:y+h, x:x+w]
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    
                    # Perform emotion recognition
                    emotion, confidence = emotion_predictor.predict(face_pil)
                    current_emotion = f'{emotion} ({confidence:.2f})'
        
        # Display current emotion in the top-left corner
        cv2.putText(display_img, f'Emotion: {current_emotion}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Display prompt message
        cv2.putText(display_img, "Press 's' to save faces", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Display FPS in the top-right corner
        text_size = cv2.getTextSize(f'FPS: {fps:.2f}', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = display_img.shape[1] - text_size[0] - 10
        cv2.putText(display_img, f'FPS: {fps:.2f}', (text_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show the result
        cv2.imshow('Face Detection', display_img)
        
        # Increment frame counter
        frame_counter += 1
        
        # Detect key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to exit
            break
        elif key == ord('s') and results.detections:  # Press 's' to save faces
            for i, detection in enumerate(results.detections):
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Extract face region from the original image
                x, y, w, h = bbox
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(w + 2*padding, iw - x)
                h = min(h + 2*padding, ih - y)
                face_img = img[y:y+h, x:x+w]  # Use original image instead of display_img
                
                # Generate a timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f'face_{timestamp}_{i}.jpg'
                filepath = os.path.join(save_dir, filename)
                
                # Save the face image
                cv2.imwrite(filepath, face_img)
                print(f"Face saved to: {filepath}")

# Release resources
cap.release()
cv2.destroyAllWindows()
