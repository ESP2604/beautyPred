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
import excelTool
from openpyxl.drawing.image import Image as OpenpyxlImage
import argparse
import MyImageTool

# 1. 检查GPU是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def loadModel():
        # 加载预训练的 EfficientNet 模型
    model = EfficientNet.from_pretrained('efficientnet-b0')  
    # 修改最后一层以适应回归任务
    model._fc = nn.Linear(model._fc.in_features, 1)
    model.load_state_dict(torch.load('models/efficient2_model_epoch_15.pth'))
    model.to(device)
    model.eval()
    return model

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

model = loadModel()
yunet = createFaceDetectorYN()

transform = transforms.Compose([
   transforms.Resize(256),
      transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def expand_coords(coords):
    """
    根据给定的扩展比例放大坐标
    :param coords: 包含四个元素的列表或元组，格式为 [x, y, width, height]
    :return: 放大后的新坐标 [new_x, new_y, new_w, new_h]
    """
        # 假设 coords 是一个包含四个元素的列表或元组，格式为 [x, y, width, height]
    x = coords[0]
    y = coords[1]
    w = coords[2] 
    h = coords[3] 
    new_x = max(0, (x - int(w * 0.2) ))  # 确保坐标不会是负数
    new_y = max(0, (y - int(h * 0.4) ))  # 确保坐标不会是负数
    new_w = int( w * 1.5) 
    new_h = int( h * 1.5)
    return new_x, new_y, new_w, new_h
def paddingReSetRect(output,coords, start_x ,start_y):

    new_x, new_y, new_w, new_h = expand_coords(coords)
    if max(0,new_x - start_x) == 0 : new_x = 0
    if max(0,new_y - start_y) == 0 : new_y = 0
    # 绘制矩形框
    cv2.rectangle(output, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

def facerect(output,coords):
        # 假设 coords 是一个包含四个元素的列表或元组，格式为 [x, y, width, height]
    new_x, new_y, new_w, new_h = expand_coords(coords)
    # 绘制矩形框
    cv2.rectangle(output, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

def visualize(image, faces):
    output = image.copy()
    list = []
    for idx, face in enumerate(faces):
        coords = face[:-1].astype(np.int32)
        # Draw face bounding box
        facerect(output, coords)
        
        new_x, new_y, new_w, new_h = expand_coords(coords)
        tmp = {
        'face_image':image[ new_y:new_y  + new_h, new_x:new_x + new_w],
            'coords':coords,
           'new_x':new_x, 
           'new_y':new_y,
           'new_w':new_w,
           'new_h':new_h
        }
        
        list.append(tmp)
        # Draw landmarks
        cv2.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
        cv2.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
        cv2.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
        cv2.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
        cv2.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
        return output, list

def printScore(image, score):
    output = image.copy()
    height, width = output.shape[:2]
     # Put score
    cv2.putText(output, 'Score:{:.4f}'.format(score), ( 0, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 0, 0) )
    return output


def pre(img):
    # 将 NumPy 数组转换为 PIL 图像
    image_pil = Image.fromarray(img)
    image_tensor =transform(image_pil).to(device)
    image_tensor =image_tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor).cpu().item()
        return prediction

def detect_and_score_faces(image):
    _, faces = yunet.detect(image)  # faces: None, or nx15 np.array
    if(faces is None): return None,None,None
    vis_image,faceImgList = visualize(image, faces)

    # 对每张脸进行评分
    score = 0
    for faceItem in faceImgList:
        score = pre(faceItem['face_image'])
        vis_image = printScore(vis_image,score)
    # 返回包含评分的图像和分数
    return vis_image, score, faceImgList


# 人臉識別 -> 臉部打分
def process_image(image):
    # 判断脸在哪
    yunet.setInputSize((image.shape[1], image.shape[0]))
    return detect_and_score_faces(image)

# filePath 檔案路徑
# count 最重試次數
def reTrySave(filePath, count):
    reTry = 0
    while reTry < count:
        try:
            filename = os.path.basename(filePath).replace(".xlsx", "")
            parentdir = os.path.dirname(filePath)
            # 保存工作簿
            wb.save(f'{os.path.join(parentdir, filename)}({reTry}).xlsx')
            break  # 如果保存成功，退出循环
        except PermissionError:
            reTry += 1  # 增加重试次数
        except Exception as e:
            print(f"保存失败：{e}")
            return  # 如果遇到其他异常，停止重试并退出

    if reTry >= count:
        print("重试次数已达上限，保存失败。")

def writeExcel(data,i):
    # 写入分数和图像名称
    ws[f'A{i}'] = data.score
    ws[f'B{i}'] = data.image_name
    ws[f'F{i}'] = data.padding_score
    # 插入图像
    if os.path.exists(data.image_path):
        excelTool.insertImage(ws, data.image_path,f'C{data.row}', data.row )
    # 插入头像
    if os.path.exists(data.avatar_path):
        excelTool.insertImage(ws, data.avatar_path,f'D{data.row}' , data.row)

    if os.path.exists(data.padding_avatar_path):
        excelTool.insertImage(ws, data.padding_avatar_path,f'E{data.row}' , data.row)

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
wb = None
ws = None
if os.path.exists(filename):
    wb = load_workbook(filename)
else:
    wb = Workbook()
# 检查工作簿中是否存在名为 "sheetname" 的工作表
if sheet_name in wb.sheetnames:
    # print("工作表已经存在")
    ws = wb[sheet_name]
else:
    # print("工作表不存在")
    ws = wb.create_sheet(sheet_name)
i = 1
imagePaths =  glob.glob(folder_path + '/*.jpg') + glob.glob(folder_path + '/*.png') +  glob.glob(folder_path + '/*.jpeg')

# 遍历文件夹中的所有图片
for img_file in imagePaths:
     # 读取图像
    image = cv2.imread(img_file)
    # 圖片 加上padding
    # 判斷圖片
    vis_image, score, faceImgList = process_image(image)
  
    
    padding_image, start_y , new_height, start_x ,  new_width = MyImageTool.create_centered_resized_image(image)
    padding_vis_image, padding_score , faceImgList = process_image(padding_image)

    if(vis_image is None  or padding_vis_image is None ): continue 
    # padding_vis_image = image.copy()
    # 通常只有一張臉
    # paddingReSetRect(padding_vis_image,faceImgList[0]['coords'], start_x, start_y)

    # 将结果图像保存
    save_path = os.path.join('TestImage/output/', os.path.basename(img_file))
    # save_path=""
    cv2.imwrite(save_path, vis_image)
    # 将结果图像保存
    padding_save_path = os.path.join('TestImage/output/', f'padding_{os.path.basename(img_file)}')

    cv2.imwrite(padding_save_path, padding_vis_image)

    # score = padding_score
    data = excelTool.MyExcelData()
    data.score = score
    data.padding_score = padding_score
    data.image_path = img_file
    data.image_name = os.path.basename(img_file)
    data.avatar_path = save_path
    data.padding_avatar_path = padding_save_path
    data.row = i
    writeExcel(data,i)
    i+=1
    if(i >limit): break
    print(f'{i}/{limit}分数[1~5]: {score}')
    # cv2.imshow('xx', vis_image)
    # cv2.waitKey(0)
reTrySave("output.xlsx",  10)

