import cv2
import numpy as np
from PIL import Image
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision.transforms as transforms
import io

class AvatarImage:
    def __init__(self, image,coords) -> None:
        self.coords = coords
        self.originalImage = image.copy()
        self.processedImage = image.copy()
        self.infoImage = image.copy()
        self.score = 0
        
        # 裁減大頭貼
        new_x, new_y, new_w, new_h = self.expand_coords()
        self.faceImage = self.originalImage[ new_y:new_y  + new_h, new_x:new_x + new_w]
        # "原始圖片"翻譯成英文是"Original Image"。
        # "加工圖片"翻譯成英文是"Processed Image"。
    def printScore(self, score):
        height, width = self.infoImage.shape[:2]
        # Put score
        cv2.putText(self.infoImage, 'Score:{:.4f}'.format(score), ( 0, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0) )
        return self
    # 函數來計算兩點之間的角度
    def find_angle_between_eyes(p1, p2):
        deltaX = p2[0] - p1[0]
        deltaY = p2[1] - p1[1]
        angleInDegrees = np.arctan2(deltaY, deltaX) * 180 / np.pi
        return angleInDegrees
    
    def correct_face_tilt(self):
        """矯正圖像中臉部的傾斜"""
        img = self.faceImage
        
        # 使用眼睛的位置計算傾斜角度
        left_eye = (self.coords[4], self.coords[5])
        right_eye = (self.coords[6], self.coords[7])
        angle = AvatarImage.find_angle_between_eyes(left_eye, right_eye)
        
        # 計算臉部中心，用於圖像旋轉
        center_of_face = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center_of_face, angle, 1)
        self.faceImage = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
     

    def paddingReSetRect( self, start_x ,start_y):
        new_x, new_y, new_w, new_h = self.expand_coords()
        if max(0,new_x - start_x) == 0 : new_x = 0
        if max(0,new_y - start_y) == 0 : new_y = 0
        # 绘制矩形框
        cv2.rectangle(self.infoImage, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
        return self

    def facerect(self):
        """
            標記臉部框框
        """
        # 假设 coords 是一个包含四个元素的列表或元组，格式为 [x, y, width, height]
        new_x, new_y, new_w, new_h = self.expand_coords()
        # 绘制矩形框
        cv2.rectangle(self.infoImage, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
        return self

    def markFeatures(self):
        '''
            標記五官特徵
        '''
        # 標記五官
        # 紅色左眼
        cv2.circle(self.infoImage, (self.coords[4], self.coords[5]), 2, (255, 0, 0), 2)
        # 藍色右眼
        cv2.circle(self.infoImage, (self.coords[6], self.coords[7]), 2, (0, 0, 255), 2)
        # 綠色鼻子
        cv2.circle(self.infoImage, (self.coords[8], self.coords[9]), 2, (0, 255, 0), 2)
        # 紅色左臉
        cv2.circle(self.infoImage, (self.coords[10], self.coords[11]), 2, (255, 0, 255), 2)
        # 黃色右臉
        cv2.circle(self.infoImage, (self.coords[12], self.coords[13]), 2, (0, 255, 255), 2)
        return self
    def expand_coords(self):
        """
        根据给定的扩展比例放大坐标
        :param coords: 包含四个元素的列表或元组，格式为 [x, y, width, height]
        :return: 放大后的新坐标 [new_x, new_y, new_w, new_h]
        """
            # 假设 coords 是一个包含四个元素的列表或元组，格式为 [x, y, width, height]
        x = self.coords[0]
        y = self.coords[1]
        w = self.coords[2] 
        h = self.coords[3] 
        width= self.infoImage.shape[0]
        height = self.infoImage.shape[1]
        # 計算可以用於裁剪的最大正方形大小
        square_size = min([width - x, x, y, height - y])

        # 確保square_size為正，並且在圖像邊界內
        square_size = max(min(square_size, w, h), 0)
        new_x= max(x - square_size, 0)
        new_y= max(y - square_size, 0)
        new_w=  (x + w + square_size) - max(x - square_size, 0) 
        new_h=  (y + h + square_size) - max(y - square_size, 0) 
        return int(new_x), int(new_y), int(new_w), int(new_h)

class CropFace:
    def __init__(self):
        self.yunet = CropFace.createFaceDetectorYN()
    
    def createFaceDetectorYN(modelStr = '.\\face_detection_yunet_2022mar.onnx'):
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
        return yunet  
    def setInputSize(self, w, h):
        self.yunet.setInputSize((w, h))

    def detect(self,image):
     
        self.yunet.setInputSize((image.shape[1], image.shape[0]))
  
        _, faces = self.yunet.detect(image)  # faces: None, or nx15 np.array
        
        
        if(faces is None): return []
        list = []
        for idx, face in enumerate(faces):
            coords = face[:-1].astype(np.int32)
            list.append(AvatarImage(image, coords))
        return list

class FaceRating:
    """
        # 读取图像
        image = cv2.imread(img_file)
        # 圖片 加上padding
        # 判斷圖片
        actorImages = cropFace.detect(image)
        # 要標記的東西
        for item in actorImages:
            # 捕捉到的臉部標記
            item.facerect()
        if len(actorImages) > 0: 
            original_images = [actor_image.originalImage for actor_image in actorImages]
            scores = faceRating.pre(original_images)
            print(scores)
            if not isinstance(scores, (list, np.ndarray)):
                scores = [scores]

            # 现在可以安全地迭代scores了
            for idx, score in enumerate(scores):
                actorImages[idx].score = score
    """
    def __init__(self):
        # 1. 检查GPU是否可用
        # if torch.cuda.is_available():
            # self.device = torch.device("cuda")
        # else:
        self.device = torch.device("cpu")
        self.model = self.loadModel()
        self.transform = transforms.Compose([
        transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def loadModel(self):
        # 加载预训练的 EfficientNet 模型
        model = EfficientNet.from_pretrained('efficientnet-b0')  
        # 修改最后一层以适应回归任务
        model._fc = nn.Linear(model._fc.in_features, 1)
        model.load_state_dict(torch.load('models/efficient2_model_epoch_15.pth'))
        model.to(self.device)
        model.eval()
        return model


    def pre(self, imgs):
        # 统一处理单张图片和图片批量
        if isinstance(imgs, np.ndarray):  # 检查是否为单张图片
            imgs = [imgs]  # 将单张图片转换为列表
        
        images_pil = [Image.fromarray(img) for img in imgs]  # 将每个 NumPy 数组转换为 PIL 图像
        images_tensor = torch.stack([self.transform(image) for image in images_pil]).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(images_tensor).cpu().numpy().flatten()
            
        if len(predictions) == 1:
            return predictions[0]  # 如果是单张图片，返回单个预测值
        return predictions  # 如果是图片批量，返回预测值数组

    def preBetch(self, imgs):
        # 适用于批量预测的方法
        images_pil = [Image.fromarray(img) for img in imgs]  # 将每个 NumPy 数组转换为 PIL 图像
        images_tensor = torch.stack([self.transform(image) for image in images_pil]).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(images_tensor).cpu().numpy().flatten()
            return predictions
# Predict single image
def crop_image(image, cols, rows):
    
    # 获取图像的宽度和高度
    width, height = image.size
    
    # 计算每块图像的宽度和高度
    col_width = width // cols
    row_height = height // rows
    
    # 存储切割后的图像块
    cropped_images = []
    
    # 切割图像
    for row in range(rows):
        for col in range(cols):
            left = col * col_width
            top = row * row_height
            right = left + col_width
            bottom = top + row_height
            cropped_image = image.crop((left, top, right, bottom))
            cropped_images.append(cropped_image)
    
    return cropped_images


  


def crop_image_into_parts(file, cols, rows):
    '''
        return: cropped_images
    '''
    # 将图像数据转换为 PIL 图像对象
    image = Image.open(io.BytesIO(file.read()))
    pil_cropped_images = crop_image(image, cols, rows)
    # 将每个PIL图像对象转换为NumPy数组
    # cropped_images = [np.array(img) for img in pil_cropped_images]
    cropped_images = [np.array(img)[:, :, ::-1] for img in pil_cropped_images]  # 在这里进行通道转换
    return cropped_images
async def async_crop_image_into_parts(file, cols, rows):
     # 将图像数据转换为 PIL 图像对象
    image = Image.open(io.BytesIO(await file.read()))
    return crop_image(image, cols, rows)

def faceRatingPre(faceRating, images):
    if len(images) > 0: 
        original_images = [actor_image.originalImage for actor_image in images]
        scores = faceRating.pre(original_images)
        print(scores)
        if not isinstance(scores, (list, np.ndarray)):
            scores = [scores]

        # 现在可以安全地迭代scores了
        for idx, score in enumerate(scores):
            images[idx].score = score