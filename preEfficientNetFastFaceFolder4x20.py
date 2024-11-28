import cv2
import numpy as np
from PIL import Image
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision.transforms as transforms
import glob
import os
from openpyxl import Workbook, load_workbook
from excelTool import ExcelSaver
import argparse
import MyImageTool
from FaceRatingTool import FaceRating, CropFace

def get_top_faces(face_scores, face_images, min_score=3, top_n=5):
    """
    Returns the top N faces based on face scores greater than a specified minimum score.
    If there are fewer than N faces, returns as many as available.

    :param face_scores: List or array of face scores.
    :param face_images: List of corresponding face images.
    :param min_score: Minimum score to filter faces.
    :param top_n: Number of top faces to return.
    :return: A tuple with two lists: top_faces_scores and top_faces_images.
    """
    # Filter out faces with scores less than the minimum threshold
    valid_faces = [(score, image) for score, image in zip(face_scores, face_images) if score > min_score]
    
    # Sort the valid faces by score (descending order)
    valid_faces.sort(key=lambda x: x[0], reverse=True)
    
    # Get the top N faces (or fewer if less than N valid faces)
    top_faces = valid_faces[:top_n]

    # Extract the scores and images of the top faces
    top_faces_scores = [face[0] for face in top_faces]
    top_faces_images = [face[1] for face in top_faces]

    return top_faces_scores, top_faces_images

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description='人脸打分并输出excel')

# 添加命令行参数
parser.add_argument('--excel', nargs='?', default='output1.xlsx', help='输出excel路径，如果有则写入excel的其他分页，默认output.xlsx')
parser.add_argument('--sheetname', nargs='?', default='Sheet', help='excel 分页名称，默认Sheet')
parser.add_argument('--source', nargs='?', default='TestImage', help='图片文件夹，默认TestImage')
parser.add_argument('--score', nargs='?', type=int, default=1, help='分数[1~5]，大于输入的值才添加进excel，默认1')
parser.add_argument('--limit', nargs='?', type=int, default=5000, help='最大处理图片数量，默认200')

# 解析命令行参数
args = parser.parse_args()
limit = args.limit

# 设定文件夹路径
folder_path = args.source

# 创建工作簿文件和sheet名
filename = args.excel
sheet_name = args.sheetname

# 初始化计数器，图片路径列表，图像处理工具类
imageIdx = 1
# imagePaths = glob.glob(folder_path + '/*.jpg') + glob.glob(folder_path + '/*.png') + glob.glob(folder_path + '/*.jpeg')
imagePaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
limit = min(limit, len(imagePaths))  # 确保最多只处理limit数量的图片

# 初始化图像处理工具实例
cropFace = CropFace()  # 用于人脸检测的工具类
faceRating = FaceRating()  # 用于人脸评分的工具类
excelSaver = ExcelSaver()  # 用于操作Excel文件的工具类

# 遍历文件夹中的所有图片
for img_file in imagePaths:
    # image_path = os.path.normpath(img_file)
    # 读取图像
    # image = cv2.imread(img_file)
    image = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
    if image is None:
        # 顯示錯誤的路徑
        print(f'{img_file}: load error')
        continue
    # 获取图片的尺寸
    height, width = image.shape[:2]
    row_count = 4  # 分割为4行
    col_count = 20  # 分割为20列
    sub_height = height // col_count  # 每块的高度
    sub_width = width // row_count   # 每块的宽度
    
    # 存储包含人脸的评分图片
    face_scores = []
    face_images = []
  
    # 对图像进行切割，获取每一个子图
    for i in range(row_count):
        for j in range(col_count):
            # 切割出一个子图
            start_y = i * sub_height
            start_x = j * sub_width
            sub_image = image[start_y:start_y + sub_height, start_x:start_x + sub_width]
            
            # 进行人脸检测
            actorImages = cropFace.detect(sub_image)
            
            # 如果没有检测到人脸或检测到多个面孔，跳过该图像
            if len(actorImages) == 0 or len(actorImages) > 1:
                continue

            # 对检测到的每个人脸进行处理
            for item in actorImages:
                # 标记人脸区域
                item.facerect()
                # 矫正人脸的角度（如头部偏斜等）
                item.correct_face_tilt()
            
            # 使用FaceRating工具类对图像进行评分
            if len(actorImages) > 0:
                faceImage = [actor_image.faceImage for actor_image in actorImages]
                scores = faceRating.pre(faceImage)
                
                # 确保scores为列表格式
                if not isinstance(scores, (list, np.ndarray)):
                    scores = [scores]

                # 将评分结果赋给每个人脸对象
                for idx, score in enumerate(scores):
                    actorImages[idx].score = score

                # 获取人脸图像和评分
                for actorImage in actorImages:
                    face_scores.append(actorImage.score)
                    face_images.append(actorImage.faceImage)
    
    if len(face_images) == 0:
        continue

    padding_image_face_scores = []
    padding_image_face_images = []
    # 对图像进行切割，获取每一个子图
    for i in range(row_count):
        for j in range(col_count):
            # 切割出一个子图
            start_y = i * sub_height
            start_x = j * sub_width
            sub_image = image[start_y:start_y + sub_height, start_x:start_x + sub_width]
            if sub_image is  None or sub_image.size == 0:
                continue
            # 给图像加上padding并重新居中（使用自定义工具）
            padding_image, start_y , new_height, start_x , new_width = MyImageTool.create_centered_resized_image(sub_image)
    
            # 进行人脸检测
            paddingActorImages = cropFace.detect(padding_image)

            # 如果没有检测到人脸或检测到多个面孔，跳过该图像
            if len(paddingActorImages) == 0 or len(paddingActorImages) > 1:
                continue

            # 对检测到的每个人脸进行处理
            for item in paddingActorImages:
                # 标记人脸区域
                item.facerect()
                # 矫正人脸的角度（如头部偏斜等）
                item.correct_face_tilt()
            
            # 使用FaceRating工具类对图像进行评分
            if len(paddingActorImages) > 0:
                faceImage = [actor_image.faceImage for actor_image in paddingActorImages]
                scores = faceRating.pre(faceImage)
                
                # 确保scores为列表格式
                if not isinstance(scores, (list, np.ndarray)):
                    scores = [scores]

                # 将评分结果赋给每个人脸对象
                for idx, score in enumerate(scores):
                    paddingActorImages[idx].score = score

                # 获取人脸图像和评分
                for actorImage in paddingActorImages:
                    padding_image_face_scores.append(actorImage.score)
                    padding_image_face_images.append(actorImage.faceImage)
   
    if len(padding_image_face_scores) == 0:
        continue
    # 选择评分最高的人脸图像
    if face_scores:
        max_score_idx = np.argmax(face_scores)
        best_face_image = face_images[max_score_idx]
        best_score = face_scores[max_score_idx]

        scores5, face5 =  get_top_faces(face_scores, face_images, min_score=3, top_n=5)

        max_score_idx = np.argmax(padding_image_face_scores)
        padding_image_best_face_image = padding_image_face_images[max_score_idx]
        padding_image_best_score = padding_image_face_scores[max_score_idx]

        padd_scores5, padd_face5 =  get_top_faces(padding_image_face_scores, padding_image_face_images, min_score=3, top_n=5)

        # 将最佳评分图像信息和评分保存到excel
        index = imageIdx
        data = [
            {'column': 'A', 'row': index, 'value': f'http://10.0.0.3:8080/player/index.html?hashPath={MyImageTool.extract_filename_without_extension(img_file)}'},
            {'column': 'B', 'row': index, 'value': img_file},  # 原始图像文件名
            {'column': 'C', 'row': index, 'value': best_face_image},  # 最佳评分人脸图像
            {'column': 'C', 'row': index, 'value': best_score},
            {'column': 'D', 'row': index, 'value': padding_image_best_face_image},  # 最佳评分人脸图像
            {'column': 'D', 'row': index, 'value': padding_image_best_score},
            {'column': 'E', 'row': index, 'value': f'=IFERROR(AVERAGEIF(G{index}:Z{index}, "<>"""), MAX(C{index},D{index}))'},
        ]
        
        for i in range(len(scores5)):
            data.append({'column': chr(ord('G') + i), 'row': index, 'value': face5[i]})
            data.append({'column': chr(ord('G') + i), 'row': index, 'value': scores5[i]})
        
        for i in range(len(padd_scores5)):
            data.append({'column': chr(ord('G') + i + len(scores5)), 'row': index, 'value': padd_face5[i]})
            data.append({'column': chr(ord('G') + i + len(scores5)), 'row': index, 'value': f'padding_score{padd_scores5[i]}'})

       
        
        print(f'{index}分数[1~5]: 最佳评分图像分数:{best_score:.2f}')
        
        # 将数据写入Excel文件
        excelSaver.writeExcel(data, index)
        imageIdx += 1
    
    # 如果处理数量超过限制，则停止处理
    if imageIdx > limit:
        break

# 保存Excel文件
excelSaver.reTrySave(10)  # 保存Excel文件，并尝试最多10次
# MobileFaceNet Sface
"""
from mobilefacenet import MobileFaceNet
import torch
import time

if __name__ == '__main__':
    filename = 'weights/mobilefacenet.pt'
    print('loading {}...'.format(filename))
    start = time.time()
    model = MobileFaceNet()
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    print('elapsed {} sec'.format(time.time() - start))
    print(model)

    output_onnx = 'weights/MobileFaceNet.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3, 112, 112)  # Example input tensor

    torch.onnx.export(model, inputs, output_onnx, export_params=True, verbose=False,
                      input_names=input_names, output_names=output_names, opset_version=10)


import onnxruntime as ort
import cv2
import numpy as np

# Load the ONNX model
onnx_model_path = 'weights/MobileFaceNet.onnx'
session = ort.InferenceSession(onnx_model_path)

# Function to preprocess the image: resize and normalize
def preprocess_image(image_path, target_size=(112, 112)):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    
    # Convert BGR (OpenCV format) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize the image
    img = img.astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1] range
    
    # Change the shape from (H, W, C) to (1, C, H, W) for ONNX
    img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W)
    img = np.expand_dims(img, axis=0)   # Add batch dimension (1, C, H, W)
    
    return img

# Load and preprocess input image
image_path = 'path_to_your_face_image.jpg'  # Replace with your image path
img = preprocess_image(image_path)

# Run inference on the ONNX model
inputs = {session.get_inputs()[0].name: img}
outputs = session.run(None, inputs)

# The output will contain face embeddings or recognition features
# The result is usually a tensor of face embeddings (size: 1 x N), where N is the number of features
face_embeddings = outputs[0]
print("Face Embeddings:", face_embeddings)

# Optional: Use these embeddings for face matching/comparison
# (You can compare embeddings to check if two faces match, e.g., using cosine similarity)


from sklearn.metrics.pairwise import cosine_similarity

def compare_embeddings(embedding1, embedding2):
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

# Example: Compare query face embedding with known face embedding
query_face_embedding = face_embeddings  # Embedding from the query face image
known_face_embedding = np.array([0.1, 0.2, 0.3])  # Example known face embedding

similarity_score = compare_embeddings(query_face_embedding, known_face_embedding)
print(f"Cosine Similarity: {similarity_score}")
"""
