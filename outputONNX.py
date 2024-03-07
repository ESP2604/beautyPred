import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from FaceRatingTool import FaceRating , CropFace, faceRatingPre ,crop_image_into_parts
# 假设 faceRating 是已经创建和初始化的 FaceRating 类实例
faceRating = FaceRating()

# 创建一个符合模型输入尺寸的虚拟输入
# 这个尺寸应该与您训练模型时使用的尺寸匹配
dummy_input = torch.randn(1, 3, 224, 224).to(faceRating.device)

# 指定输出ONNX文件的路径
output_onnx_file = 'face_rating_model.onnx'

# 导出模型
torch.onnx.export(faceRating.model,              # 要导出的模型
                  dummy_input,                   # 模型的虚拟输入
                  output_onnx_file,              # 输出文件的路径
                  export_params=True,            # 导出模型的参数
                  opset_version=11,              # 指定ONNX版本
                  do_constant_folding=True,      # 是否执行常量折叠优化
                  input_names=['input'],         # 输入层的名字
                  output_names=['output'],       # 输出层的名字
                  dynamic_axes={'input': {0: 'batch_size'},    # 批次大小的动态轴
                                'output': {0: 'batch_size'}})

print(f"模型已导出到 {output_onnx_file}")