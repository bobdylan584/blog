---
date: 2025-03-19 05:44:06
title: 图像分类_02_AlexNet
categories: [CV, 图像分类, AlexNet]
tag: AI_Module
---

## AlexNet网络结构

2012年，AlexNet横空出世，该模型的名字源于论文第一作者的姓名Alex Krizhevsky 。AlexNet使用了8层卷积神经网络，以很大的优势赢得了ImageNet 2012图像识别挑战赛。它首次证明了学习到的特征可以超越手工设计的特征，从而一举打破计算机视觉研究的方向。

在pytorch中构建AlexNet模型，继承自类nn.Module，实现init和forward方法即可，具体如下所示：

```
from torch import nn

# 创建AlexNet网络结构
class Alexnet(nn.Module):
    # 初始化过程，网络层的定义,指明输入数据的通道数和输出的类别个数
    def __init__(self, in_dim, n_class):
        super().__init__()
        # 卷积部分
        self.conv = nn.Sequential(
            # 卷积层：96个卷积核，卷积核为11*11，步幅为4：(3,227,227) -> (96,55,55)
            nn.Conv2d(in_channels=in_dim,
                      out_channels=96,
                      kernel_size=11,
                      stride=4,
                      padding=0),
            # 激活函数relu
            nn.ReLU(True),
            # 池化:窗口大小为3*3、步幅为2：(96,55,55)->(96,27,27)           
            nn.MaxPool2d(3, 2),
            # 卷积层：256个卷积核，卷积核为5*5，步幅为1，padding为2：(96,27,27) -> (256,27,27)
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            # 激活函数relu
            nn.ReLU(True),
            # 池化:窗口大小为3*3、步幅为2：(256,27,27) -> (256,13,13)
            nn.MaxPool2d(3, 2),
            # 卷积层：384个卷积核，卷积核为3*3，步幅为1，padding为1：(256,13,13) -> (384,13,13)
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            # 激活函数relu
            nn.ReLU(True),
            # 卷积层：384个卷积核，卷积核为3*3，步幅为1，padding为1：(384,13,13) -> (384,13,13)
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            # 激活函数是relu
            nn.ReLU(True),
            # 卷积层：256个卷积核，卷积核为3*3，步幅为1，padding为1:(384,13,13) -> (256,13,13)
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            # 激活函数relu
            nn.ReLU(True),
            # 池化:窗口大小为3*3、步幅为2:(256,13,13) -> (256,6,6)
            nn.MaxPool2d(3, 2))
        # 全连接层部分
        self.fc = nn.Sequential(
            # 全连接层:4096个神经元
            nn.Linear(9216, 4096), 
            # 激活函数relu
            nn.ReLU(True),
            # 随机失活
            nn.Dropout(0.5), 
            # 全连接层:4096个神经元
            nn.Linear(4096, 4096),
            # 激活函数relu
            nn.ReLU(True), 
            # 随机失活
            nn.Dropout(0.5),
            # 全连接层:n_class个神经元
            nn.Linear(4096, n_class))

    def forward(self, x):
        # 输入数据送入卷积部分进行处理
        x = self.conv(x)
        #将特征图抻成一维向量
        x = x.view(x.size(0), -1)  
        # 全连接层获取输出结果
        output = self.fc(x)
        return output
```

我们构造一个高和宽均为227的单通道数据样本来看一下模型的架构：

```
from torchsummary import summary
import torch
# 模型实例化
net = Alexnet(3,1000)
# 通过net.summay()查看网络的形状
summary(model=net,input_size=(3,227,227),batch_size=1,device='cpu')
```

网络架构如下：

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [1, 96, 55, 55]          34,944
              ReLU-2            [1, 96, 55, 55]               0
         MaxPool2d-3            [1, 96, 27, 27]               0
            Conv2d-4           [1, 256, 27, 27]         614,656
              ReLU-5           [1, 256, 27, 27]               0
         MaxPool2d-6           [1, 256, 13, 13]               0
            Conv2d-7           [1, 384, 13, 13]         885,120
              ReLU-8           [1, 384, 13, 13]               0
            Conv2d-9           [1, 384, 13, 13]       1,327,488
             ReLU-10           [1, 384, 13, 13]               0
           Conv2d-11           [1, 256, 13, 13]         884,992
             ReLU-12           [1, 256, 13, 13]               0
        MaxPool2d-13             [1, 256, 6, 6]               0
           Linear-14                  [1, 4096]      37,752,832
             ReLU-15                  [1, 4096]               0
          Dropout-16                  [1, 4096]               0
           Linear-17                  [1, 4096]      16,781,312
             ReLU-18                  [1, 4096]               0
          Dropout-19                  [1, 4096]               0
           Linear-20                  [1, 1000]       4,097,000
================================================================
Total params: 62,378,344
Trainable params: 62,378,344
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.59
Forward/backward pass size (MB): 11.05
Params size (MB): 237.95
Estimated Total Size (MB): 249.59
----------------------------------------------------------------
```

## 2.鲜花种类识别

AlexNet使用ImageNet数据集进行训练，但因为ImageNet数据集较大训练时间较长，我们使用前面的介绍的鲜花分类数据集来演示AlexNet。读取数据的时将图像高和宽扩大到AlexNet使用的图像高和宽227。

###  数据读取

首先获取数据,为了适应AlexNet对输入数据尺寸的要求，将图像大小调整到227x227的大小，并使用DataLoader进行批次数据的读取：

```
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
# 指定批次大小
batch_size = 2
# 指定数据集路径
flower_train_path = './dataset/flower_datas/train/'
flower_test_path = './dataset/flower_datas/val/'
# 先将数据转换为tensor类型，并调整数据的大小为227x227
dataset_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((227, 227))])
# 获取训练集数据和测试集数据
flower_train = ImageFolder(flower_train_path, transform=dataset_transform)
flower_test = ImageFolder(flower_test_path, transform=dataset_transform)
# 获取数据的迭代
train_loader = DataLoader(dataset=flower_train,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=flower_ttest
                         batch_size=batch_size,
                         shuffle=False)
```

为了更好的理解，我们将batch数据展示出来：

```
import matplotlib.pyplot as plt
# 遍历每个迭代的数据，将其结果展示出来
for b, (imgs, targets) in enumerate(train_loader):
    # 获取第一个batch的图像
    if b == 1:
        # 将其进行展示
        fig, axes = plt.subplots(1, 2)
        # 遍历batch中的每个图像
        for i in range(batch_size):
            # 图像显示出来
            axes[i].imshow(imgs[i].permute(1, 2, 0))
            # 设置图像标题
            axes[i].set_title(targets[i].item())
        plt.show()
    elif b > 0:
        break
        
```

2.2 模型实例化和参数设置[¶](#22)

```
# 模型实例化:输入数据3通道，进行5类的分类处理
model = Alexnet(3, 5)
# 模型训练的参数设置
# 学习率
learning_rate = 1e-3
# 训练轮数
num_epochs = 10
# 优化算法Adam = RMSProp + Momentum
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 交叉熵损失函数
loss_fn = torch.nn.CrossEntropyLoss()
```

问题记录：

### 假设220x220x1的一个图片，相当于一张不同颜色像素点的图片，为什么要3通道堆叠？

相当于宽高220x220个像素点，每个像素点有三个通道（三原色的亮度，明暗程度）RGB(red的强度，green的强度，blue的强度)

它的样子：它看起来就是黑白的。0是纯黑，255是纯白，中间是不同深浅的灰色。





它是什么：这张图片只有一个通道。这个通道里的每一个数值（0-255）只代表一个信息：像素的亮度（明暗程度）。

它的样子：它看起来就是黑白的。0是纯黑，255是纯白，中间是不同深浅的灰色。

数据格式：它的形状是 (高度, 宽度) 或 (高度, 宽度, 1)。最后一个维度1就表示“通道数为1”。

如何处理：我们直接对这个矩阵进行操作。不需要也不应该把它复制成3份。