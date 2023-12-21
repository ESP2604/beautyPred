
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn


import os
from efficientnet_pytorch import EfficientNet

from PIL import Image

from tqdm import tqdm
# 監控
from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('runs/experiment_name')
batch_size = 16


def data(pathStr = 'data/test.txt'):
    # 讀取 test.txt 文件
    with open(pathStr, "r") as f:
        lines = f.readlines()

    # 將每一行分割成文件名和標記
    data = [line.strip().split() for line in lines]

    # 將文件名和標記存儲到列表中
    file_names = [d[0] for d in data]
    labels = [float(d[1]) for d in data]


    # 構建文件路徑
    file_paths = [os.path.join(".\\data\\Images\\", f) for f in file_names]
    return file_paths, labels

# 定义数据预处理/转换
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  

class MyDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        # label = self.labels[index]
       
        label = torch.Tensor([float(self.labels[index])])
        # 使用Pillow库打开图像
        image = Image.open(file_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.file_paths)


# 1. 检查GPU是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 加载预训练的 EfficientNet 模型
model = EfficientNet.from_pretrained('efficientnet-b0')  
# 修改最后一层以适应回归任务
model._fc = nn.Linear(model._fc.in_features, 1)
model.to(device)

criterion = nn.MSELoss()
# optimizer_alexnet = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,  weight_decay=5e-4)
optimizer_alexnet = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler_alexnet = optim.lr_scheduler.StepLR(optimizer_alexnet, step_size=500, gamma=0.1)

file_paths, labels = data("data/train.txt")
# 載入數據集
train_dataset = MyDataset(file_paths , labels, transform=transform)# 這裡替換為您要使用的數據集
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size )#, shuffle=True) #,pin_memory=True)

file_paths, labels = data("data/test.txt")
test_dataset = MyDataset(file_paths , labels ,transform=transform)# 這裡替換為您要使用的數據集
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)# shuffle=False) #,pin_memory=True)

num_epochs = 20

# 训练模型
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch + 1}', dynamic_ncols=True)
    for batch_data, batch_labels in train_loader:
        batch_labels = batch_labels.to(device)
        batch_data = batch_data.to(device)

        optimizer_alexnet.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer_alexnet.step()
        # 更新進度條描述
        train_bar.set_description(f'Epoch {epoch + 1} Loss: {loss.item():.4f}')
        train_bar.update(1)  # 更新進度條
        # 計入 loss
        train_loss += loss.item()
    # 更新
    lr_scheduler_alexnet.step()
    # 更新 BAR
    train_bar.close()
       # 计算平均训练损失并记录
    train_loss /= len(train_loader)
    writer.add_scalar('Loss/Train', train_loss, epoch)

    # 在测试集上评估模型
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        total = 0
        test_bar = tqdm(test_loader, total=len(test_loader), desc=f'Epoch {epoch + 1} Test', dynamic_ncols=True)
        for batch_data, batch_labels in test_loader:
            batch_labels = batch_labels.to(device)
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            test_bar.set_description(f'Epoch {epoch + 1} MSE Loss: {loss.item():.4f}')  # 打印MSE损失
            test_bar.update(1)  # 更新進度條
             # 計入 loss
            test_loss += loss.item()
            # 计算平均测试损失并记录
        test_loss /= len(test_loader)
        writer.add_scalar('Loss/Test', test_loss, epoch)
       
       
 
        test_bar.close()

    # 每5个epoch保存一次模型
    if (epoch + 1) % 5 == 0:
        save_dir = 'models'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), f'./models/efficient2_model_epoch_{epoch + 1}.pth')
    # 保存模型权重和结构
torch.save(model.state_dict(), 'model.pth')
print('訓練完成')

writer.close()