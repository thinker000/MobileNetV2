import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        
        # 修改初始卷积层以适应28x28的输入
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            self._make_layer(32, 64, stride=1),  # 深度可分离卷积块
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2),
            self._make_layer(512, 1024, stride=1),
        )
        
        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器（全连接层）
        self.classifier = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        """深度可分离卷积层块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 确保输入数据是4D张量，形状为 [batch_size, 1, 28, 28]
        if len(x.shape) == 2:  # 如果输入是2D的，则重新调整为4D
            x = x.view(-1, 1, 28, 28)  # 调整输入的形状

        x = self.features(x)  # 前向传播特征提取
        x = self.global_avg_pool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展开为一维向量
        x = self.classifier(x)  # 分类器
        return x

# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # 先调整为32x32，以便与网络兼容
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

'''
MNIST简单全连接模型
'''
class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor



'''
Cifar简单全连接模型
'''
class Cifar_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor

'''
MNIST简单卷积神经网络模型
'''
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义每一层模型
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3*3*64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, inputs):
        # 构造模型
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = F.relu(self.conv3(tensor))     
        tensor = tensor.view(-1, 3*3*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

'''
cifar简单卷积神经网络模型
'''
class Cifar_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义每一层模型
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8*8*128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, inputs):
        # 构造模型
        tensor = inputs.view(-1, 3, 32, 32)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = F.relu(self.conv3(tensor))  
        # print(tensor.shape)
        # raise(1)
        tensor = tensor.view(-1, 8*8*128)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

class MyCifar_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义每一层模型
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批量归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 批量归一化
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # 批量归一化
        
        # 全连接层
        self.fc1 = nn.Linear(8*8*128, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)  # Dropout层

    def forward(self, inputs):
        # 这一步是将输入确保为 (batch_size, 3, 32, 32) 的四维张量
        tensor = inputs.view(-1, 3, 32, 32)
        
        # 构造模型
        tensor = F.relu(self.bn1(self.conv1(tensor)))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.bn2(self.conv2(tensor)))
        tensor = self.pool2(tensor)
        tensor = F.relu(self.bn3(self.conv3(tensor)))  
        
        # 展平为一维向量
        tensor = tensor.view(-1, 8*8*128)
        
        # 全连接层
        tensor = F.relu(self.fc1(tensor))
        tensor = self.dropout(tensor)  # 添加 Dropout
        tensor = self.fc2(tensor)
        
        return tensor

'''
resnet卷积神经网络模型
'''
class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        tensor = self.conv1(inputs)
        tensor = F.relu(self.bn1(tensor))
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
        return F.relu(inputs + tensor)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, input):
        extra_x = self.extra(input)
        tensor = self.conv1(input)
        tensor = F.relu(self.bn1(tensor))
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
        return F.relu(extra_x + tensor)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 3, 32, 32)
        tensor = self.conv1(tensor)
        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer4(tensor)
        tensor = self.avgpool(tensor)
        tensor = tensor.reshape(tensor.shape[0], -1)
        tensor = self.fc(tensor)
        return tensor

