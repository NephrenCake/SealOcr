import torch
import torch.nn as nn
import math
class SPP_Layer(nn.Module):
    def __init__(self, num_levels= 2, pool_type='max_pool'):
        super(SPP_Layer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type
    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))
            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

class SPP_Net1(nn.Module):
    def __init__(self, num_classes=3):  # 这里的类别数
        super(SPP_Net1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # (224+2*2-11)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55-3)/2+1=27
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # (27+2*2-5)/1+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-3)/2+1=13
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
        )
        self.pool = SPP_Layer(2,'max_pool')
        self.classifier = nn.Sequential( #为了简化模型，去除两个全连接层
            nn.Linear(1280, num_classes),
        )
    def forward(self, x):  # 定义前向传播方法
        x = self.features(x)
        # print(x.shape)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SPP_Net2(nn.Module):
    def __init__(self, num_classes=3):  # 这里的类别数
        super(SPP_Net2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),  # (224+2*2-11)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55-3)/2+1=27
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # (27+2*2-5)/1+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-3)/2+1=13
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),  # 13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
        )
        self.pool = SPP_Layer(2,'max_pool')
        self.classifier = nn.Sequential( #为了简化模型，去除两个全连接层
            nn.Linear(960, num_classes),
        )
    def forward(self, x):  # 定义前向传播方法
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SPP_Net3(nn.Module):
    def __init__(self, num_classes=3):  # 这里的类别数
        super(SPP_Net3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),  # (224+2*2-11)/4+1=55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55-3)/2+1=27
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # (27+2*2-5)/1+1=27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-3)/2+1=13
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),  # 13+1*2-3)/1+1=13
            nn.ReLU(inplace=True),
        )
        self.pool = SPP_Layer(2,'max_pool')
        self.classifier = nn.Sequential( #为了简化模型，去除两个全连接层
            nn.Linear(960, num_classes),
        )
    def forward(self, x):  # 定义前向传播方法
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x