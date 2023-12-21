import cv2

import numpy as np

from PIL import Image
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision.transforms as transforms

# 1. 检查GPU是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    # 加载预训练的 EfficientNet 模型
model = EfficientNet.from_pretrained('efficientnet-b0')  
# 修改最后一层以适应回归任务
model._fc = nn.Linear(model._fc.in_features, 1)
model.load_state_dict(torch.load('models/efficient2_model_epoch_15.pth'))
model.to(device)

model.eval()

transform = transforms.Compose([
   transforms.Resize(256),
      transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def center(output,coords):

        # 假设 coords 是一个包含四个元素的列表或元组，格式为 [x, y, width, height]
    x = coords[0]
    y = coords[1]
    w = coords[2] 
    h = coords[3] 
    

    # 计算放大后的新坐标
    expand_size = 0 # 想要放大的尺寸
    new_x = max(0, (x - int( w * 0.2) ))  # 确保坐标不会是负数
    new_y = max(0, (y - int(h*0.4) ))  # 确保坐标不会是负数
    new_w = int( w * 1.5) 
    new_h = int( h * 1.5)

    # 绘制矩形框
    cv2.rectangle(output, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

def visualize(image, faces):
    output = image.copy()
    list = []
    for idx, face in enumerate(faces):
        coords = face[:-1].astype(np.int32)
        # Draw face bounding box
        # cv2.rectangle(output, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 2)
        center(output, coords)
        x = coords[0]
        y = coords[1]
        w = coords[2] 
        h = coords[3] 
        new_x = max(0, (x - int( w * 0.2) ))  # 确保坐标不会是负数
        new_y = max(0, (y - int(h*0.4) ))  # 确保坐标不会是负数
        new_w = int( w * 1.5) 
        new_h = int( h * 1.5)
        list.append(image[ new_y:new_y  + new_h, new_x:new_x + new_w])
        # Draw landmarks
        cv2.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
        cv2.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
        cv2.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
        cv2.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
        cv2.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
        # Put score
        cv2.putText(output, '{:.4f}'.format(face[-1]), (coords[0], coords[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 0))
        return output, list
def printScore(image, score):
    output = image.copy()
    height, width = output.shape[:2]
     # Put score
    cv2.putText(output, 'Score:{:.4f}'.format(score), ( 0, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (201, 206, 189) )
    return output

def pre(img):
    # 将 NumPy 数组转换为 PIL 图像
    image_pil = Image.fromarray(img)
    image_tensor =transform(image_pil).to(device)
    image_tensor =image_tensor.unsqueeze(0)
    with torch.no_grad():
      
        prediction = model(image_tensor).cpu().item()
        return prediction

# 读取图像
imgPathStr = 'data\\c9217933dbf195ccc4e0bc0f95ac8297.jpg'
imgPathStr = "data\\4409861PH.jpg"
imgPathStr = 'data\\2.jpg'

image = cv2.imread(imgPathStr)
modelStr = '.\\face_detection_yunet_2022mar.onnx'
# modelStr = '.\\centerface_bnmerged.onnx'
score_threshold = 0.85
nms_threshold = 0.35
backend = cv2.dnn.DNN_BACKEND_DEFAULT
target = cv2.dnn.DNN_TARGET_CPU
# Instantiate yunet
  
yunet = cv2.FaceDetectorYN.create(
    model=modelStr,
    config='',
    input_size=(320, 320),
    score_threshold=score_threshold,
    nms_threshold=nms_threshold,
    top_k=5000,
    backend_id=backend,
    target_id=target
)


yunet.setInputSize((image.shape[1], image.shape[0]))
_, faces = yunet.detect(image)  # faces: None, or nx15 np.array
vis_image, faceImgList = visualize(image, faces)


vis = True
if vis:
    for faceImg in faceImgList:
        s = pre(faceImg)
        print(f'分數[1~5]:{s}')
    # cv2.namedWindow('xx', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('xx', printScore(vis_image, s))
    cv2.waitKey(0)


# https://github.com/serengil/retinaface 臉部識別
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 加载您的评分模型
# # 加载预训练的 EfficientNet 模型
# model = EfficientNet.from_pretrained('efficientnet-b0')  
# # 修改最后一层以适应回归任务
# model._fc = nn.Linear(model._fc.in_features, 1)
# model.to(device)
# model.load_state_dict(torch.load('models\efficient2_model_epoch_20.pth'))  # 加载模型权重
# model.to(device)  # 将模型移动到 GPU 或 CPU 上
# model.eval()

# # 图像预处理转换
# transform = transforms.Compose([
#    transforms.Resize(256),
#       transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


