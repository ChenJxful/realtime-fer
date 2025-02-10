import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
# 添加一个空的RecorderMeter1类来解决加载问题
class RecorderMeter1(object):
    pass
# 添加一个空的RecorderMeter1类来解决加载问题
class RecorderMeter(object):
    pass
class ONNXEmotionPredictor:
    def __init__(self, onnx_path):
        """
        初始化ONNX推理器
        """
        # 创建ONNX运行时会话
        self.session = onnxruntime.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 表情标签映射
        self.emotion_labels = {
            0: "Surprise",
            1: "Happy", 
            2: "Sad",
            3: "Angry",
            4: "Neutral"
        }
    
    def predict(self, image):
        """
        对图片进行表情识别
        
        Args:
            image: PIL.Image对象或图片路径
            
        Returns:
            predicted_emotion: 预测的表情标签
            confidence: 预测的置信度
        """
        # 处理输入图片
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 预处理图片
        img_tensor = self.transform(image)
        img_numpy = img_tensor.unsqueeze(0).numpy()
        
        # 获取输入输出名称
        input_name = self.session.get_inputs()[0].name
        
        # 进行推理
        outputs = self.session.run(None, {input_name: img_numpy})
        probabilities = outputs[0][0]
        
        # 使用softmax获取概率
        exp_scores = np.exp(probabilities)
        probs = exp_scores / np.sum(exp_scores)
        
        # 获取预测结果
        predicted_idx = np.argmax(probs)
        confidence = probs[predicted_idx]
        predicted_emotion = self.emotion_labels[predicted_idx]
        
        return predicted_emotion, float(confidence)

# 使用示例
if __name__ == "__main__":
    # 初始化ONNX预测器
    predictor = ONNXEmotionPredictor("./emotion_model.onnx")
    
    # 进行预测
    image_path = "./happy.jpg"
    
    # 测试推理速度
    num_tests = 10
    total_time = 0.0
    
    for _ in range(num_tests):
        start_time = time.time()
        emotion, confidence = predictor.predict(image_path)
        end_time = time.time()
        
        inference_time = end_time - start_time
        total_time += inference_time
        print(f"单次推理时间: {inference_time:.3f}秒")
    
    average_time = total_time / num_tests
    print(f"平均推理时间: {average_time:.3f}秒")
    
    # 打印最后一次预测结果
    print(f"预测的表情: {emotion}")
    print(f"置信度: {confidence:.2f}") 
