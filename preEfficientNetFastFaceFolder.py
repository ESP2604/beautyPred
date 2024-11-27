import cv2
import numpy as np
from PIL import Image
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision.transforms as transforms
import glob
import os
from openpyxl import Workbook ,load_workbook
from excelTool import ExcelSaver
# from openpyxl.drawing.image import Image as OpenpyxlImage
import argparse
import MyImageTool
from FaceRatingTool import FaceRating , CropFace

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description='人臉打分輸出excel')

# 添加命令行参数，可以传入不同的值来定制程序的行为
parser.add_argument('--excel', nargs='?', default='output1.xlsx', help='輸出excel路徑，如果有則寫入excel的其他分頁，預設為output.xlsx')
parser.add_argument('--sheetname', nargs='?', default='Sheet', help='excel 分頁名稱，預設為sheet')
# parser.add_argument('--source', nargs='?' , default='TestImage', help='圖片資料夾，預設TestImage')

parser.add_argument('--source', nargs='?' , default='TestImage\katarinabluu', help='圖片資料夾，預設TestImage')
parser.add_argument('--score', nargs='?', type=int, default=1, help='分數[1~5]，大於輸入的值才添加進excel，預設為1')
parser.add_argument('--limit', nargs='?', type=int, default=200, help='最大處理圖片數量 預設為200')

# 解析命令行参数
args = parser.parse_args()
limit = args.limit

# 设定文件夹路径（通过命令行参数）
folder_path = args.source

# 创建工作簿文件和sheet名
filename = args.excel
sheet_name = args.sheetname

# 初始化计数器，图片路径列表，图像处理工具类
i = 1
imagePaths =  glob.glob(folder_path + '/*.jpg') + glob.glob(folder_path + '/*.png') +  glob.glob(folder_path + '/*.jpeg')
limit = min(limit, len(imagePaths))  # 确保最多只处理limit数量的图片

# 初始化图像处理工具实例
cropFace = CropFace()  # 用于人脸检测的工具类
faceRating = FaceRating()  # 用于人脸评分的工具类
excelSaver = ExcelSaver()  # 用于操作Excel文件的工具类

# 遍历文件夹中的所有图片
for img_file in imagePaths:
    # 读取图像
    image = cv2.imread(img_file)
    
    # 图像进行加上padding，准备检测人脸
    actorImages = cropFace.detect(image)
    
    # 如果没有检测到人脸或检测到多个面孔，跳过该图像
    if(len(actorImages) == 0 or len(actorImages) > 1):
        continue

    # 对检测到的每个人脸进行处理
    for item in actorImages:
        # 标记人脸区域
        item.facerect()
        # 矫正人脸的角度（如头部偏斜等）
        item.correct_face_tilt()
    
    # 如果检测到至少一张人脸
    if len(actorImages) > 0: 
        # 获取原始人脸图像列表
        original_images = [actor_image.originalImage for actor_image in actorImages]
        
        # 使用FaceRating工具类对图像进行打分
        scores = faceRating.pre(original_images)
        # print(scores)
      
        # 确保scores为列表格式
        if not isinstance(scores, (list, np.ndarray)):
            scores = [scores]

        # 将评分结果赋给每个人脸对象
        for idx, score in enumerate(scores):
            actorImages[idx].score = score

        faceImage = [actorImage.faceImage for actorImage in actorImages]
        faceImageScores = faceRating.pre(faceImage)
        # print(faceImageScores)
        # 确保scores为列表格式
        if not isinstance(faceImageScores, (list, np.ndarray)):
            faceImageScores = [faceImageScores]

        # 将评分结果赋给每个人脸对象
        for idx, faceImageScore in enumerate(faceImageScores):
            actorImages[idx].faceImageScore = faceImageScore
            
    
    # 给图像加上padding并重新居中（使用自定义工具）
    padding_image, start_y , new_height, start_x , new_width = MyImageTool.create_centered_resized_image(image)
    
    # 对加上padding的图像进行人脸检测
    paddingActorImages = cropFace.detect(padding_image)
    if(len(paddingActorImages) == 0 or len(paddingActorImages) > 1): 
        continue
    
    # 对检测到的人脸进行处理
    for item in paddingActorImages:
        item.facerect()

    # 如果检测到至少一张人脸
    if len(paddingActorImages) > 0: 
        # 获取加上padding后的原始人脸图像列表
        original_images = [actor_image.originalImage for actor_image in paddingActorImages]
        # 使用FaceRating工具类对加上padding的图像进行打分
        scores = faceRating.pre(original_images)
        
        # 确保scores为列表格式
        if not isinstance(scores, (list, np.ndarray)):
            scores = [scores]

        # 将评分结果赋给每个人脸对象
        for idx, score in enumerate(scores):
            paddingActorImages[idx].score = score

    
 


    # 获取原始和加上padding的图像及其评分
    actorImage, paddingActorImage = actorImages[0], paddingActorImages[0]
    
    # 记录原始图像评分
    original_score = actorImage.score
    
    # 如果人脸矫正后评分发生了变化，则需要重新评分并更新
    corrected_score = paddingActorImage.score  # 矫正后的评分
    
    # 设置当前图像的行号
    index =  i 

    # 准备要写入Excel的数据
    data = [ 
        {'column':'A','row':index, 'value': original_score},  # 原始图像评分
        {'column':'B', 'row':index,'value':actorImage.infoImage },  # 原始图像文件名
        {'column':'C', 'row':index,'value':actorImage.faceImage },  # 原始人脸图像
        {'column':'D', 'row':index,'value':actorImage.faceImageScore },  # 原始人脸图像
      
        {'column':'E', 'row':index,'value':paddingActorImage.infoImage },  # 矫正后图像件名
        {'column':'F', 'row':index,'value': corrected_score },  # 矫正后图像评分
    ]
    
    # 打印当前图像的评分信息
    print(f'{index}分数[1~5]: original分數:{original_score} 矯正過後的圖片分數:{actorImage.faceImageScore} padding分數:{corrected_score}')
    
    # 将数据写入Excel文件
    excelSaver.writeExcel(data, index)
    
    i += 1
    
    # 如果处理数量超过限制，则停止处理
    if(i > limit): 
        break
    
# 保存Excel文件
excelSaver.reTrySave(10)  # 保存Excel文件，并尝试最多10次
