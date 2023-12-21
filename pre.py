import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import Nets  # 导入您的模型定义

from retinaface import RetinaFace
# https://github.com/serengil/retinaface 臉部識別
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载您的评分模型
model = Nets.AlexNet()  # 替换为您的模型定义
model.load_state_dict(torch.load('models\model_epoch_20.pth'))  # 加载模型权重
model.to(device)  # 将模型移动到 GPU 或 CPU 上
model.eval()

# 图像预处理转换
transform = transforms.Compose([
   transforms.Resize(256),
      transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 读取图像
imgPathStr = 'data\\c9217933dbf195ccc4e0bc0f95ac8297.jpg'
imgPathStr = "data\\4409861PH.jpg"
imgPathStr = 'data\\2.jpg'

faces = RetinaFace.extract_faces(img_path = imgPathStr, align = False)
for face in faces:

    plt.imshow(face)
 

    image_pil  = Image.fromarray(face)
    image_tensor =transform(image_pil).to(device)
    image_tensor =image_tensor.unsqueeze(0)
    img = image_tensor
    # 使用模型评分
    with torch.no_grad():
        output = model(img)

    # 获取评分结果
    score = output.item()
    print(f"score: {score}")
    plt.show()
# 处理每个检测到的人脸

    # {
#     "face_1": {
#         "score": 0.9993440508842468,
#         "facial_area": [155, 81, 434, 443],
#         "landmarks": {
#           "right_eye": [257.82974, 209.64787],
#           "left_eye": [374.93427, 251.78687],
#           "nose": [303.4773, 299.91144],
#           "mouth_right": [228.37329, 338.73193],
#           "mouth_left": [320.21982, 374.58798]
#         }
#   }
# }

    # x, y, w, h = face.left(), face.top(), face.width(), face.height()
 
  




# 显示带有人脸评分的图像
# cv2.imshow('Image with Scores', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
