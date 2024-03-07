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
# 定义要检查的目录路径

# 創建 ArgumentParser 對象
parser = argparse.ArgumentParser(description='人臉打分輸出excel')

# 添加參數，並為部分參數設置預設值
parser.add_argument('--excel', nargs='?', default='output1.xlsx', help='輸出excel路徑，如果有則寫入excel的其他分頁，預設為output.xlsx')
parser.add_argument('--sheetname', nargs='?', default='Sheet', help='excel 分頁名稱，預設為sheet')
parser.add_argument('--source', nargs='?' , default='TestImage', help='圖片資料夾，預設TestImage')
parser.add_argument('--score', nargs='?', type=int, default=1, help='分數[1~5]，大於輸入的值才添加進excel，預設為1')
parser.add_argument('--limit', nargs='?', type=int, default=200, help='最大處理圖片數量 預設為200')
# 解析參數
args = parser.parse_args()
limit = args.limit
# 设定文件夹路径
folder_path = args.source

# 创建一个工作簿
filename = args.excel
sheet_name = args.sheetname

i = 1
imagePaths =  glob.glob(folder_path + '/*.jpg') + glob.glob(folder_path + '/*.png') +  glob.glob(folder_path + '/*.jpeg')
limit = min(limit, len(imagePaths))
cropFace = CropFace()
faceRating = FaceRating()
excelSaver = ExcelSaver()
# 遍历文件夹中的所有图片
for img_file in imagePaths:
     # 读取图像
    image = cv2.imread(img_file)
    # 圖片 加上padding
    # 判斷圖片
    actorImages = cropFace.detect(image)
    if(len(actorImages) == 0 or len(actorImages) > 1) :continue
    # 要標記的東西
    for item in actorImages:
        # 捕捉到的臉部標記
        item.facerect()
        # item.markFeatures()
        # 把臉橋正
        item.correct_face_tilt()
    if len(actorImages) > 0: 
        original_images = [actor_image.originalImage for actor_image in actorImages]
        scores = faceRating.pre(original_images)
        print(scores)
        if not isinstance(scores, (list, np.ndarray)):
            scores = [scores]

        # 现在可以安全地迭代scores了
        for idx, score in enumerate(scores):
            actorImages[idx].score = score
    
    padding_image, start_y , new_height, start_x ,  new_width = MyImageTool.create_centered_resized_image(image)
    

    paddingActorImages = cropFace.detect(padding_image)
    if(len(paddingActorImages) == 0 or len(paddingActorImages) > 1) :continue
    for item in paddingActorImages:
        item.facerect()

    if len(paddingActorImages) > 0: 
        original_images = [actor_image.originalImage for actor_image in paddingActorImages]
        scores = faceRating.pre(original_images)
        # 确保scores是一个列表
        if not isinstance(scores, (list, np.ndarray)):
            scores = [scores]

        # 现在可以安全地迭代scores了
        for idx, score in enumerate(scores):
            paddingActorImages[idx].score = score



   
    actorImage, paddingActorImage =  actorImages[0], paddingActorImages[0]
    index =  i 
    data =[ 
    {'column':'A','row':index, 'value': actorImage.score},
        {'column':'B', 'row':index,'value':actorImage.infoImage },
    {'column':'C', 'row':index,'value':actorImage.faceImage },
        {'column':'D', 'row':index,'value':paddingActorImage.score },
        {'column':'D', 'row':index,'value':paddingActorImage.infoImage },
]
    print(f'{index}分数[1~5]: original:{actorImage.score} padding:{actorImage.score}')
    excelSaver.writeExcel(data, index)
    
    i+=1
    if(i >limit): break
    
excelSaver.reTrySave(10)