# 项目简介

## 项目名称

**项目名称：** BeautyPrediction

## 项目描述

**项目描述：** BeautyPrediction 是一个基于深度学习的人脸美观度评分工具。它使用预训练的神经网络模型来分析人脸图像，并为每张人脸图像分配一个美观度评分。该工具可以用于自动评估人脸图像的美观度，并帮助用户选择最具吸引力的图像。

本項目使用[SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)對efficientnet-b0進行訓練

# 项目描述

 **项目描述：** BeautyPrediction 是一个基于深度学习的人脸美观度评分工具。该工具使用预训练的神经网络模型 efficientnet-b0，并针对美观度评分任务进行了训练。其主要目的是自动评估人脸图像的美观度，并帮助用户选择最具吸引力的图像。这个项目的训练数据来自 [SCUT-FBP5500 数据集](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)，通过对大量人脸图像进行训练，模型可以有效地预测人脸的美观度。 

## 技术细节 

BeautyPrediction 使用了 efficientnet-b0 模型，这是一个轻量级但功能强大的卷积神经网络。模型经过预训练，然后通过进一步的训练来适应美观度评分任务。它具有出色的性能和泛化能力，可以在各种应用中使用。 

# 使用文档

## 安装依赖项

```bash
git clone https://github.com/yourusername/BeautyPrediction.git
cd BeautyPrediction
pip install -r requirements.txt
```

## 下载预训练模型

为了使用 BeautyPrediction 工具，你需要下载预训练的模型权重文件。你可以从以下链接下载模型权重文件：

[BeautyPrediction 模型权重](https://huggingface.co/opa2604/beauty_prediction/tree/main)

将下载的模型权重文件保存到项目的 `models` 文件夹中。

## 使用示例

下面是一个简单的使用示例，演示如何使用 BeautyPrediction 工具来评估人脸图像的美观度并将结果写入 Excel 文件：

## 运行 BeautyPrediction 工具

要运行 BeautyPrediction 工具，只需运行脚本并提供适当的命令行参数：

```bash
python preEfficientNetFastFaceFolder.py --excel "output.xlsx" --sheetname "Sheet1" --source "TestImage" --score 1 --limit 10
```



### 使用命令行参数

你可以使用以下命令行参数来自定义 BeautyPrediction 工具的行为：

- `--excel`：指定要输出的 Excel 文件路径，默认为 "output1.xlsx"。
- `--sheetname`：指定要使用的 Excel 分页名称，默认为 "Sheet"。
- `--source`：指定包含人脸图像的文件夹路径，默认为 "TestImage"。
- `--score`：指定美观度分数的阈值，只添加大于阈值的图像到 Excel，默认为 1。
- `--limit`：指定最大处理图像的数量，默认为 200。