import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from PIL import Image
import time

# 创建保存人脸图像的文件夹
save_dir = 'detected-faces'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 初始化mediapipe人脸检测
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# 初始化ONNX表情识别器
from onnx_inference import ONNXEmotionPredictor
emotion_predictor = ONNXEmotionPredictor("./emotion_model.onnx")

# 打开摄像头
cap = cv2.VideoCapture(0)

# 使用上下文管理器创建人脸检测器
with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.7) as face_detection:

    # 初始化帧计数器和当前表情
    frame_counter = 0
    current_emotion = "Unknown"
    fps = 0
    prev_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break
        
        # 计算当前时间和上一帧时间的差值
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        
        # 计算FPS
        if elapsed_time > 0:
            fps = 1 / elapsed_time
        
        # 水平翻转图像
        img = cv2.flip(img, 1)
        
        # 将BGR图像转换为RGB图像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 确保图像是连续的内存块
        img_rgb = np.ascontiguousarray(img_rgb)
        
        # 进行人脸检测
        results = face_detection.process(img_rgb)
        
        # 创建显示用的图像副本
        display_img = img.copy()
        
        # 如果检测到人脸
        if results.detections:
            for i, detection in enumerate(results.detections):
                # 获取边界框坐标
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # 在显示图像上绘制绿色矩形框
                cv2.rectangle(display_img, bbox, (0, 255, 0), 2)
                
                # 每3帧进行一次表情识别
                if frame_counter % 3 == 0:
                    # 从原始图像提取人脸区域
                    x, y, w, h = bbox
                    face_img = img[y:y+h, x:x+w]
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    
                    # 进行表情识别
                    emotion, confidence = emotion_predictor.predict(face_pil)
                    current_emotion = f'{emotion} ({confidence:.2f})'
        
        # 在左上角显示当前表情
        cv2.putText(display_img, f'Emotion: {current_emotion}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # 显示提示信息
        cv2.putText(display_img, "Press 's' to save faces", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # 显示FPS在右上角
        text_size = cv2.getTextSize(f'FPS: {fps:.2f}', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = display_img.shape[1] - text_size[0] - 10
        cv2.putText(display_img, f'FPS: {fps:.2f}', (text_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Face Detection', display_img)
        
        # 增加帧计数器
        frame_counter += 1
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按'q'键退出
            break
        elif key == ord('s') and results.detections:  # 按's'键保存人脸
            for i, detection in enumerate(results.detections):
                # 获取边界框坐标
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # 从原始图像提取人脸区域
                x, y, w, h = bbox
                # 添加一些边距
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(w + 2*padding, iw - x)
                h = min(h + 2*padding, ih - y)
                face_img = img[y:y+h, x:x+w]  # 使用原始图像而不是display_img
                
                # 生成带时间戳的文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f'face_{timestamp}_{i}.jpg'
                filepath = os.path.join(save_dir, filename)
                
                # 保存人脸图像
                cv2.imwrite(filepath, face_img)
                print(f"Face saved to: {filepath}")

# 释放资源
cap.release()
cv2.destroyAllWindows()
